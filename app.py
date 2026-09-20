from flask import Flask, request, jsonify, render_template

from pypdf import PdfReader

from werkzeug.utils import secure_filename

import requests

import markdown

from pymongo import MongoClient

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

import json

import re

import traceback

import tempfile

from faster_whisper import WhisperModel

import os

import uuid

import numpy as np

from datetime import datetime, timezone

from groq import Groq





app = Flask(__name__)





load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "qwen/qwen3.8-27b"


def ask_groq(messages, temperature=0.2, max_tokens=1200, json_mode=False):
    kwargs = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "stream": False,
        "reasoning_effort": "none",
        "reasoning_format": "hidden",
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = groq_client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""

    if not content.strip():
        raise RuntimeError("Groq returned an empty response.")

    return content.strip()





UPLOAD_FOLDER = os.path.join(

    os.path.dirname(os.path.abspath(__file__)),

    "uploads"

)





os.makedirs(

    UPLOAD_FOLDER,

    exist_ok=True

)





app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER





mongo_uri = os.getenv("MONGODB_URI")





client = MongoClient(

    mongo_uri,

    serverSelectionTimeoutMS=10000

)





db = client["EduGPT"]





chats_collection = db["conversations"]



pdf_chunks_collection = db["pdf_chunks"]





chat_id = str(uuid.uuid4())





conversation_history = []





embedding_model = SentenceTransformer(

    "all-MiniLM-L6-v2"

)



# Local speech-to-text model (loaded on first voice request)

whisper_model = None



def get_whisper_model():

    global whisper_model

    if whisper_model is None:

        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    return whisper_model





def split_text_into_chunks(

    text,

    chunk_size=1000,

    chunk_overlap=200

):



    text = text.strip()





    if not text:

        return []





    chunks = []





    start = 0



    text_length = len(text)





    while start < text_length:



        end = start + chunk_size





        chunk = text[start:end].strip()





        if chunk:



            chunks.append(chunk)





        start += chunk_size - chunk_overlap





    return chunks





def cosine_similarity(

    vector_a,

    vector_b

):



    vector_a = np.array(

        vector_a

    )



    vector_b = np.array(

        vector_b

    )





    denominator = (

        np.linalg.norm(vector_a)

        *

        np.linalg.norm(vector_b)

    )





    if denominator == 0:



        return 0.0





    return float(

        np.dot(

            vector_a,

            vector_b

        )

        /

        denominator

    )





def search_pdf_chunks(

    question,

    top_k=5,

    filename=None,

    similarity_threshold=0.25

):

    question_embedding = embedding_model.encode(question)

    query = {"filename": filename} if filename else {}

    stored_chunks = pdf_chunks_collection.find(

        query,

        {"_id": 0, "filename": 1, "chunk_index": 1, "text": 1, "embedding": 1}

    )

    results = []

    for chunk in stored_chunks:

        stored_embedding = chunk.get("embedding")

        if not stored_embedding:

            continue

        similarity = cosine_similarity(question_embedding, stored_embedding)

        if similarity >= similarity_threshold:

            results.append({

                "filename": chunk.get("filename", ""),

                "chunk_index": chunk.get("chunk_index", 0),

                "text": chunk.get("text", ""),

                "similarity": similarity

            })

    results.sort(key=lambda item: item["similarity"], reverse=True)

    return results[:top_k]





def classify_pdf_relevance(question, chunks):

    """Second-stage RAG check: decide whether retrieved chunks are relevant."""

    if not chunks:

        return False

    chunk_text = "\n\n--- CHUNK ---\n\n".join(

        f"Filename: {c.get('filename', '')}\nChunk: {c.get('chunk_index', 0)}\nText: {c.get('text', '')}"

        for c in chunks

    )

    prompt = f"""You are a strict relevance classifier.



User question:

{question}



Retrieved PDF chunks:

{chunk_text}



Return JSON only: {{"relevant": true}} or {{"relevant": false}}.

Return true only if the chunks contain information meaningfully relevant to the question.

A shared word alone is not enough. General questions unrelated to the PDF are false.

Do not answer the question; only classify relevance."""

    content = ask_groq(
        [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=80,
        json_mode=True,
    )

    if "</think>" in content:

        content = content.split("</think>", 1)[1].strip()

    try:

        return bool(json.loads(content).get("relevant", False))

    except Exception:

        return False





@app.route("/")

def home():



    return render_template(

        "index.html"

    )







def summarize_pdf_with_groq(chunks, filename):

    """

    Summarize a PDF using Groq in batches so large PDFs do not

    have to be sent to the model in one huge request.

    """



    if not chunks:

        raise ValueError(

            "No PDF text is available to summarize."

        )



    batch_size = 5

    partial_summaries = []



    for start in range(0, len(chunks), batch_size):



        batch = chunks[start:start + batch_size]



        batch_text = "\n\n---\n\n".join(

            batch

        )



        prompt = (

            "You are EduGPT, an educational AI assistant.\n\n"

            f"Create a clear study summary of the following part of "

            f"the PDF '{filename}'.\n\n"

            "Use ONLY the provided PDF text. Do not invent facts.\n"

            "Keep important names, events, concepts, and details.\n"

            "Write concise bullet points and short sections.\n\n"

            "PDF TEXT:\n\n"

            + batch_text

        )



        answer = ask_groq(
            [
                {
                    "role": "system",
                    "content": (
                        "You summarize educational documents "
                        "faithfully and clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_tokens=1800,
        )



        if "</think>" in answer:

            answer = answer.split(

                "</think>",

                1

            )[1].strip()



        partial_summaries.append(

            answer.strip()

        )



    # If there was only one batch, it is already the final summary.

    if len(partial_summaries) == 1:

        return partial_summaries[0]



    combined = "\n\n---\n\n".join(

        partial_summaries

    )



    final_prompt = (

        "You are EduGPT, an educational AI assistant.\n\n"

        f"Create one final study summary for the PDF '{filename}' "

        "from the partial summaries below.\n\n"

        "Use ONLY the information in these partial summaries.\n"

        "Do not invent or add outside facts.\n"

        "Remove repetition and organize the result clearly.\n\n"

        "Use this structure when supported by the document:\n"

        "## Overview\n"

        "## Main Points\n"

        "## Important Details\n"

        "## Key Takeaway\n\n"

        "PARTIAL SUMMARIES:\n\n"

        + combined

    )



    final_answer = ask_groq(
        [
            {
                "role": "system",
                "content": (
                    "You create accurate, student-friendly "
                    "summaries from supplied text."
                ),
            },
            {
                "role": "user",
                "content": final_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1800,
    )



    if "</think>" in final_answer:

        final_answer = final_answer.split(

            "</think>",

            1

        )[1].strip()



    return final_answer.strip()





@app.route(

    "/summarize-pdf",

    methods=["POST"]

)

def summarize_pdf():



    data = request.get_json(

        silent=True

    ) or {}



    filename = data.get(

        "filename",

        ""

    ).strip()



    if not filename:

        return jsonify({

            "response": "Please upload a PDF first."

        }), 400



    try:



        stored_chunks = pdf_chunks_collection.find(

            {

                "filename": filename

            },

            {

                "_id": 0,

                "chunk_index": 1,

                "text": 1

            }

        ).sort(

            "chunk_index",

            1

        )



        chunks = [

            item.get("text", "")

            for item in stored_chunks

            if item.get("text", "").strip()

        ]



        if not chunks:

            return jsonify({

                "response": (

                    f"No processed text was found for "

                    f"'{filename}'. Please upload the PDF again."

                )

            }), 404



        print("\n========== PDF SUMMARY ==========")

        print("PDF:", filename)

        print("Chunks:", len(chunks))

        print("=================================\n")



        summary = summarize_pdf_with_groq(

            chunks,

            filename

        )



        summary_html = markdown.markdown(

            summary,

            extensions=[

                "tables",

                "fenced_code"

            ]

        )



        return jsonify({

            "response": summary_html,

            "filename": filename,

            "chunk_count": len(chunks)

        })



    except requests.exceptions.ConnectionError:



        return jsonify({

            "response": (

                "Cannot connect to Groq. "

                "Make sure Groq is running."

            )

        }), 500



    except requests.exceptions.Timeout:



        return jsonify({

            "response": (

                "Groq took too long to create the summary. "

                "Please try again."

            )

        }), 500



    except Exception as e:



        print("\n========== SUMMARY ERROR ==========")

        print("ERROR:", repr(e))

        traceback.print_exc()

        print("===================================\n")



        return jsonify({

            "response": (

                "Summary error: "

                + str(e)

            )

        }), 500







def _normalize_mcq_items(parsed, question_count):

    """Accept several reasonable Groq JSON shapes and normalize them."""

    if isinstance(parsed, dict):

        for key in ("questions", "mcqs", "items", "data"):

            if isinstance(parsed.get(key), list):

                parsed = parsed[key]

                break

        else:

            # Sometimes the model returns {"1": {...}, "2": {...}}

            if all(isinstance(v, dict) for v in parsed.values()):

                parsed = list(parsed.values())

            else:

                parsed = [parsed]



    if not isinstance(parsed, list):

        return []



    valid = []



    for item in parsed:

        if not isinstance(item, dict):

            continue



        question = str(

            item.get("question")

            or item.get("question_text")

            or item.get("text")

            or ""

        ).strip()



        options = item.get("options") or item.get("choices") or {}

        answer = str(

            item.get("answer")

            or item.get("correct_answer")

            or item.get("correct")

            or ""

        ).strip()

        explanation = str(

            item.get("explanation")

            or item.get("reason")

            or ""

        ).strip()



        # Support options as a list: ["...", "...", "...", "..."]

        if isinstance(options, list) and len(options) >= 4:

            options = {

                "A": str(options[0]).strip(),

                "B": str(options[1]).strip(),

                "C": str(options[2]).strip(),

                "D": str(options[3]).strip(),

            }



        if not isinstance(options, dict):

            continue



        normalized = {}

        for key in ("A", "B", "C", "D"):

            value = options.get(key)

            if value is None:

                value = options.get(key.lower())

            if value is None:

                value = options.get(f"Option {key}")

            normalized[key] = str(value or "").strip()



        if not question or any(not normalized[k] for k in normalized):

            continue



        # Normalize answers such as "B", "Option B", "(B)", "B.", or full option text.

        answer_clean = answer.strip()

        answer_upper = answer_clean.upper()

        match = re.search(r"\b([ABCD])\b", answer_upper)

        if match:

            answer_letter = match.group(1)

        else:

            answer_letter = ""

            for key, value in normalized.items():

                if answer_clean.lower() == value.lower():

                    answer_letter = key

                    break



        if answer_letter not in ("A", "B", "C", "D"):

            continue



        valid.append({

            "question": question,

            "options": normalized,

            "answer": answer_letter,

            "explanation": explanation

        })



        if len(valid) >= question_count:

            break



    return valid





def _extract_json_payload(content):

    """Extract the first usable JSON object/array from Groq output."""

    text = (content or "").strip()



    if "</think>" in text:

        text = text.split("</think>", 1)[1].strip()



    # Remove common Markdown fences.

    text = re.sub(r"^\s*\`\`\`(?:json)?\s*", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s*\`\`\`\s*$", "", text).strip()



    # First try the complete response.

    try:

        return json.loads(text)

    except Exception:

        pass



    # Then find a balanced JSON object or array.

    for opening, closing in (("{", "}"), ("[", "]")):

        start = text.find(opening)

        while start >= 0:

            depth = 0

            in_string = False

            escaped = False



            for i in range(start, len(text)):

                ch = text[i]



                if in_string:

                    if escaped:

                        escaped = False

                    elif ch == "\\\\":

                        escaped = True

                    elif ch == '"':

                        in_string = False

                    continue



                if ch == '"':

                    in_string = True

                elif ch == opening:

                    depth += 1

                elif ch == closing:

                    depth -= 1

                    if depth == 0:

                        candidate = text[start:i + 1]

                        try:

                            return json.loads(candidate)

                        except Exception:

                            break



            start = text.find(opening, start + 1)



    return None





def generate_mcqs_with_groq(chunks, filename, question_count=10):

    """Generate MCQs using only the supplied PDF text."""

    if not chunks:

        raise ValueError("No PDF text is available for MCQ generation.")



    # 14 chunks from the current PDF are small enough, but keep a safe bound.

    source_text = "\n\n---\n\n".join(chunks)

    source_text = source_text[:10000]



    prompt = f"""

You are EduGPT, an educational quiz generator.



Create exactly {question_count} multiple-choice questions using ONLY the PDF text below.



STRICT RULES:

\- Do not use outside knowledge.

\- Every question must be answerable directly from the PDF.

\- Each question has exactly four options.

\- The options must be labeled exactly A, B, C, D.

\- Exactly one answer is correct.

\- "answer" MUST be exactly one of: A, B, C, D.

\- Include a short explanation.

\- Return ONLY JSON.

\- Do not include Markdown, commentary, or a thinking/reasoning section.



Return this exact JSON structure:

{{

  "questions": [

    {{

      "question": "Example question?",

      "options": {{

        "A": "First option",

        "B": "Second option",

        "C": "Third option",

        "D": "Fourth option"

      }},

      "answer": "A",

      "explanation": "Why A is correct based on the PDF."

    }}

  ]

}}



PDF: {filename}



PDF TEXT:

{source_text}

"""



    content = ask_groq(
        [
            {
                "role": "system",
                "content": "Generate structured educational MCQs from the supplied document.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1800,
        json_mode=True,
    )



    parsed = _extract_json_payload(content)

    if parsed is None:

        raise RuntimeError(

            "Groq returned text that could not be parsed as MCQ JSON."

        )



    valid_questions = _normalize_mcq_items(parsed, question_count)



    if not valid_questions:

        raise RuntimeError(

            "Groq returned JSON, but the MCQ fields were not in a usable format."

        )



    return valid_questions





@app.route(

    "/generate-mcq",

    methods=["POST"]

)

def generate_mcq():



    data = request.get_json(

        silent=True

    ) or {}



    filename = data.get(

        "filename",

        ""

    ).strip()



    try:

        requested_count = int(

            data.get("count", 10)

        )

    except (TypeError, ValueError):

        requested_count = 10



    requested_count = max(

        1,

        min(requested_count, 10)

    )



    if not filename:

        return jsonify({

            "response": "Please upload a PDF first."

        }), 400



    try:



        stored_chunks = pdf_chunks_collection.find(

            {

                "filename": filename

            },

            {

                "_id": 0,

                "chunk_index": 1,

                "text": 1

            }

        ).sort(

            "chunk_index",

            1

        )



        chunks = [

            item.get("text", "")

            for item in stored_chunks

            if item.get("text", "").strip()

        ]



        if not chunks:

            return jsonify({

                "response": (

                    f"No processed text was found for "

                    f"'{filename}'. Please upload the PDF again."

                )

            }), 404



        print("\n========== MCQ GENERATION ==========")

        print("PDF:", filename)

        print("Chunks:", len(chunks))

        print("Questions requested:", requested_count)

        print("====================================\n")



        questions = generate_mcqs_with_groq(

            chunks,

            filename,

            requested_count

        )



        return jsonify({

            "response": "MCQs generated successfully.",

            "filename": filename,

            "questions": questions

        })



    except requests.exceptions.ConnectionError:



        return jsonify({

            "response": (

                "Cannot connect to Groq. "

                "Make sure Groq is running."

            )

        }), 500



    except requests.exceptions.Timeout:



        return jsonify({

            "response": (

                "Groq took too long to generate the MCQs. "

                "Please try again."

            )

        }), 500



    except Exception as e:



        print("\n========== MCQ ERROR ==========")

        print("ERROR:", repr(e))

        traceback.print_exc()

        print("===============================\n")



        return jsonify({

            "error": (

                "MCQ generation error: "

                + str(e)

            ),

            "response": (

                "MCQ generation error: "

                + str(e)

            )

        }), 500





@app.route("/transcribe-audio", methods=["POST"])

def transcribe_audio():

    """Transcribe browser-recorded audio locally with Whisper."""

    audio = request.files.get("audio")

    if not audio:

        return jsonify({"error": "No audio file was received."}), 400



    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:

            temp_path = tmp.name

            audio.save(temp_path)



        model = get_whisper_model()

        segments, info = model.transcribe(

            temp_path,

            beam_size=5,

            vad_filter=True,

            language="en"

        )

        text = " ".join(segment.text.strip() for segment in segments).strip()



        if not text:

            return jsonify({"error": "No speech was detected. Please try again."}), 400



        return jsonify({"text": text, "language": getattr(info, "language", "en")})



    except Exception as e:

        traceback.print_exc()

        return jsonify({"error": f"Voice transcription failed: {str(e)}"}), 500

    finally:

        if temp_path:

            try:

                os.remove(temp_path)

            except OSError:

                pass





@app.route(

    "/chat",

    methods=["POST"]

)

def chat():



    global conversation_history





    data = request.get_json(

        silent=True

    ) or {}





    user_message = data.get(

        "message",

        ""

    ).strip()





    if not user_message:



        return jsonify({

            "response": "Please enter a message."

        })





    try:



        pdf_filename = data.get(

            "pdf_filename",

            ""

        ).strip()



        study_mode = bool(data.get("study_mode", False))

        study_level = str(

            data.get("study_level", "beginner")

        ).strip().lower()



        if study_level not in (

            "beginner",

            "intermediate",

            "advanced"

        ):

            study_level = "beginner"



        retrieved_chunks = search_pdf_chunks(

            user_message,

            top_k=5,

            filename=pdf_filename if pdf_filename else None,

            similarity_threshold=0.25

        )



        # Stage 2: verify semantic relevance before giving PDF text to the answer model.

        pdf_is_relevant = classify_pdf_relevance(user_message, retrieved_chunks)

        relevant_chunks = retrieved_chunks if pdf_is_relevant else []



        pdf_context = "\n\n---\n\n".join(

            "Source: " + chunk["filename"] + "\n\n" + chunk["text"]

            for chunk in relevant_chunks

        )



        system_instruction = """

You are EduGPT, an educational AI assistant.



Answer general questions using your general knowledge.

For questions about the uploaded PDF, use the supplied relevant PDF context.



IMPORTANT:

\- Never mention the PDF, retrieved chunks, RAG, similarity, or retrieval for a general question.

\- Never force a general question to become a PDF question.

\- PDF context is private reference material and should be used silently.

\- For claims about the PDF, use only the supplied PDF context.

\- Do not invent PDF characters, events, facts, or details.

\- If the user explicitly asks about the PDF and the supplied context is insufficient, say so clearly.

\- Do not invent a medical diagnosis from a short question; provide general educational information instead.

\- Keep the answer clear, accurate, educational, and concise.

"""



        if pdf_context:

            system_instruction += "\nRELEVANT PDF CONTEXT (use silently):\n\n" + pdf_context



        if study_mode:

            level_instructions = {

                "beginner": "Use very simple language. Define important terms and use an everyday example.",

                "intermediate": "Use moderate technical detail and explain relationships between ideas with an example.",

                "advanced": "Give a deeper technical explanation with important assumptions, nuances, and related concepts."

            }

            study_instruction = (

                "\n\nSTUDY MODE IS ON.\n"

                f"Student level: {study_level}.\n"

                f"{level_instructions[study_level]}\n\n"

                "Teach the student instead of only giving a short answer. "

                "When appropriate use: 1. Direct answer 2. Step-by-step explanation "

                "3. Example 4. Key points to remember 5. One short Quick Check question.\n"

                "Do not reveal hidden reasoning or chain-of-thought. "

                "Do not invent facts from the PDF. "

                "For medical topics, provide general educational information and do not diagnose."

            )

            system_instruction += study_instruction



        messages_for_model = [

            {

                "role": "system",

                "content": system_instruction

            }

        ]





        messages_for_model.extend(

            conversation_history

        )





        messages_for_model.append({

            "role": "user",

            "content": user_message

        })





        answer = ask_groq(
            messages_for_model,
            temperature=0.2,
            max_tokens=2000,
        )





        if "</think>" in answer:



            answer = answer.split(

                "</think>",

                1

            )[1].strip()





        conversation_history.append({

            "role": "user",

            "content": user_message

        })





        conversation_history.append({

            "role": "assistant",

            "content": answer

        })





        now = datetime.now(

            timezone.utc

        )





        chats_collection.update_one(

            {

                "chat_id": chat_id

            },

            {

                "$setOnInsert": {

                    "chat_id": chat_id,

                    "title": user_message[:50],

                    "created_at": now

                },

                "$set": {

                    "updated_at": now

                },

                "$push": {

                    "messages": {

                        "user_message": user_message,

                        "assistant_message": answer,

                        "created_at": now

                    }

                }

            },

            upsert=True

        )





        answer_html = markdown.markdown(

            answer,

            extensions=[

                "tables",

                "fenced_code"

            ]

        )





        return jsonify({

            "response": answer_html

        })





    except requests.exceptions.ConnectionError:



        return jsonify({

            "response": (

                "Cannot connect to Groq. "

                "Make sure Groq is running."

            )

        }), 500





    except requests.exceptions.Timeout:



        return jsonify({

            "response": (

                "Groq took too long to respond. "

                "Please try again."

            )

        }), 500





    except Exception as e:



        print(

            "ERROR:",

            repr(e)

        )





        return jsonify({

            "response": f"Error: {str(e)}"

        }), 500





@app.route(

    "/new-chat",

    methods=["POST"]

)

def new_chat():



    global conversation_history

    global chat_id





    conversation_history = []





    chat_id = str(

        uuid.uuid4()

    )





    return jsonify({

        "message": "New chat started.",

        "chat_id": chat_id

    })





@app.route(

    "/chat-history",

    methods=["GET"]

)

def chat_history():

    history = []



    try:

        chats = chats_collection.find(

            {},

            {

                "_id": 0,

                "chat_id": 1,

                "title": 1,

                "created_at": 1,

                "updated_at": 1

            }

        ).sort("updated_at", -1)



        for chat in chats:

            if not chat.get("chat_id"):

                continue



            history.append({

                "chat_id": str(chat.get("chat_id")),

                "title": chat.get("title") or "New Chat",

                "created_at": chat.get("created_at"),

                "updated_at": chat.get("updated_at")

            })



        return jsonify(history)



    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error": f"Could not load chat history: {str(e)}"

        }), 500





@app.route("/check-answer", methods=["POST"])

def check_answer():

    data = request.get_json(silent=True) or {}



    question = str(data.get("question", "")).strip()

    answer = str(data.get("answer", "")).strip()



    if not question:

        return jsonify({"error": "Please provide the Quick Check question."}), 400



    if not answer:

        return jsonify({"error": "Please provide an answer."}), 400



    try:

        prompt = f"""

You are checking a student's answer in an educational app.



Quick Check question:

{question}



Student answer:

{answer}



Evaluate the student's answer to the Quick Check question.



Return ONLY JSON:

{{

  "correct": true,

  "feedback": "Short helpful feedback."

}}



If the answer is correct or essentially correct, use true.

If it is wrong or substantially incomplete, use false.

Be encouraging and educational.

Do not reveal hidden reasoning.

"""



        content = ask_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            json_mode=True,
        )
        parsed = json.loads(content)



        return jsonify({

            "correct": bool(parsed.get("correct", False)),

            "feedback": str(

                parsed.get("feedback", "Answer checked.")

            )

        })



    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error": f"Answer checking failed: {str(e)}"

        }), 500





@app.route("/rename-chat/<chat_id_to_rename>", methods=["POST", "PATCH"])

def rename_chat(chat_id_to_rename):

    data = request.get_json(silent=True) or {}

    new_title = str(data.get("title", "")).strip()[:80]



    if not chat_id_to_rename:

        return jsonify({"error": "Chat ID is required."}), 400

    if not new_title:

        return jsonify({"error": "Chat title cannot be empty."}), 400



    result = chats_collection.update_one(

        {"chat_id": str(chat_id_to_rename)},

        {"$set": {"title": new_title, "updated_at": datetime.now(timezone.utc)}}

    )



    if result.matched_count == 0:

        return jsonify({"error": "Chat not found."}), 404



    return jsonify({

        "message": "Chat renamed successfully.",

        "chat_id": str(chat_id_to_rename),

        "title": new_title

    })





@app.route("/delete-chat/<chat_id_to_delete>", methods=["DELETE"])

def delete_chat(chat_id_to_delete):

    global chat_id, conversation_history



    result = chats_collection.delete_one({"chat_id": str(chat_id_to_delete)})

    if result.deleted_count == 0:

        return jsonify({"error": "Chat not found."}), 404



    if str(chat_id) == str(chat_id_to_delete):

        chat_id = str(uuid.uuid4())

        conversation_history = []



    return jsonify({

        "message": "Chat deleted successfully.",

        "chat_id": str(chat_id_to_delete)

    })





@app.route(

    "/load-chat/<chat_id_to_load>",

    methods=["GET"]

)

def load_chat(

    chat_id_to_load

):



    global chat_id

    global conversation_history





    chat = chats_collection.find_one(

        {

            "chat_id": chat_id_to_load

        },

        {

            "_id": 0,

            "messages": 1

        }

    )





    if not chat:



        return jsonify({

            "messages": []

        })





    conversation_history = []





    chat_messages = []





    for message in chat.get(

        "messages",

        []

    ):



        user_message = message.get(

            "user_message",

            ""

        )





        assistant_message = message.get(

            "assistant_message",

            ""

        )





        conversation_history.append({

            "role": "user",

            "content": user_message

        })





        conversation_history.append({

            "role": "assistant",

            "content": assistant_message

        })





        chat_messages.append({

            "user_message": user_message,

            "assistant_message": markdown.markdown(

                assistant_message,

                extensions=[

                    "tables",

                    "fenced_code"

                ]

            )

        })





    chat_id = chat_id_to_load





    return jsonify({

        "messages": chat_messages

    })





@app.route(

    "/upload-pdf",

    methods=["POST"]

)

def upload_pdf():



    if "file" not in request.files:



        return jsonify({

            "error": "No file uploaded."

        }), 400





    file = request.files["file"]





    if file.filename == "":



        return jsonify({

            "error": "No file selected."

        }), 400





    if not file.filename.lower().endswith(

        ".pdf"

    ):



        return jsonify({

            "error": "Only PDF files are allowed."

        }), 400





    filename = secure_filename(

        file.filename

    )





    file_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )





    file.save(file_path)





    try:



        reader = PdfReader(

            file_path

        )





        extracted_text = ""





        for page in reader.pages:



            text = page.extract_text()





            if text:



                extracted_text += (

                    text + "\n"

                )





        if not extracted_text.strip():



            return jsonify({

                "error": (

                    "Could not extract text "

                    "from this PDF."

                )

            }), 400





        chunks = split_text_into_chunks(

            extracted_text,

            chunk_size=1000,

            chunk_overlap=200

        )





        if not chunks:



            return jsonify({

                "error": (

                    "No text chunks were created "

                    "from this PDF."

                )

            }), 400





        embeddings = embedding_model.encode(

            chunks

        )





        pdf_chunks_collection.delete_many({

            "filename": filename

        })





        for index, chunk in enumerate(

            chunks

        ):



            pdf_chunks_collection.insert_one({



                "filename": filename,



                "chunk_index": index,



                "text": chunk,



                "embedding": embeddings[index].tolist(),



                "created_at": datetime.now(

                    timezone.utc

                )



            })





        print("\n")



        print("=" * 60)



        print(

            "PDF:",

            filename

        )



        print(

            "Pages:",

            len(reader.pages)

        )



        print(

            "Extracted characters:",

            len(extracted_text)

        )



        print(

            "Number of chunks:",

            len(chunks)

        )



        print(

            "Number of embeddings:",

            len(embeddings)

        )



        print("=" * 60)





        for index, chunk in enumerate(

            chunks[:5],

            start=1

        ):



            print(

                f"\nCHUNK {index}\n"

            )





            print(chunk)





            print(

                "\n" + "-" * 60

            )





        print("=" * 60)



        print("\n")





        return jsonify({



            "message": (

                "PDF uploaded and processed successfully."

            ),



            "filename": filename,



            "pages": len(reader.pages),



            "text_length": len(extracted_text),



            "chunk_count": len(chunks),



            "embedding_count": len(embeddings),



            "chunks": chunks



        })





    except Exception as e:



        print(

            "PDF ERROR:",

            repr(e)

        )





        return jsonify({

            "error": (

                f"Could not read PDF: {str(e)}"

            )

        }), 500





@app.route(

    "/search-pdf",

    methods=["POST"]

)

def search_pdf():



    data = request.get_json(

        silent=True

    ) or {}





    question = data.get(

        "question",

        ""

    ).strip()





    if not question:



        return jsonify({

            "error": "Please enter a question."

        }), 400





    try:



        results = search_pdf_chunks(

            question,

            top_k=3

        )





        response_results = []





        for result in results:



            response_results.append({



                "filename":

                    result["filename"],



                "chunk_index":

                    result["chunk_index"],



                "similarity":

                    round(

                        result["similarity"],

                        4

                    ),



                "text":

                    result["text"]



            })





        return jsonify({

            "question": question,

            "results": response_results

        })





    except Exception as e:



        print(

            "SEARCH ERROR:",

            repr(e)

        )





        return jsonify({

            "error": str(e)

        }), 500





if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(os.getenv("PORT", "5000")),

        debug=False

    )