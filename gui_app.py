"""
gui_app.py
----------
Yerel RAG Asistani icin masaustu pencere uygulamasi (Tkinter ile).
Python'un standart kutuphanesinde geldigi icin ekstra kurulum
gerektirmez.

Calistirma:
    python gui_app.py
"""

import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox

from foundry_local_sdk import Configuration, FoundryLocalManager

import db
import search

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "qwen2.5-0.5b"
TOP_K = 3

SYSTEM_PROMPT_TEMPLATE = (
    "Sana verilen BAGLAM (context) disina cikma. "
    "Sadece asagidaki baglamdaki bilgiyi kullanarak Turkce cevap ver. "
    "Eger cevap baglamda yoksa, 'Bu bilgi elimdeki dokumanlarda yok' de. "
    "Cevabinin sonunda hangi kaynak dosya(lar)dan yararlandigini belirt.\n\n"
    "BAGLAM:\n{context}"
)


class RagApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yerel RAG Soru-Cevap Asistani")
        self.root.geometry("700x600")
        self.root.configure(bg="#1e1e1e")

        self.embedding_client = None
        self.chat_client = None
        self.embedding_model = None
        self.chat_model = None
        self.all_chunks = []
        self.models_ready = False

        self._build_ui()
        self._load_models_in_background()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Yerel RAG Soru-Cevap Asistani",
            font=("Segoe UI", 16, "bold"),
            bg="#1e1e1e",
            fg="white",
        )
        title.pack(pady=(15, 5))

        self.status_label = tk.Label(
            self.root,
            text="Modeller yukleniyor, lutfen bekleyin...",
            font=("Segoe UI", 10),
            bg="#1e1e1e",
            fg="#ffb347",
        )
        self.status_label.pack(pady=(0, 10))

        self.chat_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="#2b2b2b",
            fg="white",
            state="disabled",
            height=22,
        )
        self.chat_area.pack(padx=15, pady=5, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(self.root, bg="#1e1e1e")
        input_frame.pack(padx=15, pady=15, fill=tk.X)

        self.entry = tk.Entry(input_frame, font=("Segoe UI", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self.entry.bind("<Return>", lambda event: self.send_question())
        self.entry.config(state="disabled")

        self.send_button = tk.Button(
            input_frame,
            text="Gonder",
            font=("Segoe UI", 11, "bold"),
            bg="#0078D4",
            fg="white",
            command=self.send_question,
            state="disabled",
        )
        self.send_button.pack(side=tk.LEFT, padx=(10, 0))

    def _append_chat(self, sender: str, text: str, color: str = "white"):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, f"{sender}: ", ("bold",))
        self.chat_area.insert(tk.END, f"{text}\n\n")
        self.chat_area.tag_config("bold", font=("Segoe UI", 11, "bold"))
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    def _load_models_in_background(self):
        thread = threading.Thread(target=self._load_models, daemon=True)
        thread.start()

    def _load_models(self):
        try:
            if db.count_chunks() == 0:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Hata",
                        "Veritabani bos! Once 'python ingest.py' calistirin.",
                    ),
                )
                return

            config = Configuration(app_name="foundry_local_rag_gui")
            FoundryLocalManager.initialize(config)
            manager = FoundryLocalManager.instance

            self.embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
            self.embedding_model.download(lambda p: None)
            self.embedding_model.load()
            self.embedding_client = self.embedding_model.get_embedding_client()

            self.chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
            self.chat_model.download(lambda p: None)
            self.chat_model.load()
            self.chat_client = self.chat_model.get_chat_client()

            self.all_chunks = db.get_all_chunks()
            self.models_ready = True

            self.root.after(0, self._on_models_ready)
        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("Hata", f"Modeller yuklenemedi:\n{e}")
            )

    def _on_models_ready(self):
        self.status_label.config(
            text=f"Hazir! Veritabaninda {len(self.all_chunks)} bilgi parcasi var.",
            fg="#4caf50",
        )
        self.entry.config(state="normal")
        self.send_button.config(state="normal")
        self.entry.focus()
        self._append_chat(
            "Asistan", "Merhaba! Dokumanlarimla ilgili sorularinizi yanitlayabilirim."
        )

    def send_question(self):
        if not self.models_ready:
            return
        query = self.entry.get().strip()
        if not query:
            return
        self.entry.delete(0, tk.END)
        self._append_chat("Siz", query)
        self.entry.config(state="disabled")
        self.send_button.config(state="disabled")
        self.status_label.config(text="Cevap hazirlaniyor...", fg="#ffb347")

        thread = threading.Thread(target=self._answer_question, args=(query,), daemon=True)
        thread.start()

    def _answer_question(self, query: str):
        try:
            query_response = self.embedding_client.generate_embedding(query)
            query_embedding = query_response.data[0].embedding

            results = search.find_relevant(query_embedding, self.all_chunks, top_k=TOP_K)
            context_lines = []
            sources = []
            for score, chunk in results:
                context_lines.append(f"- ({chunk['source']}) {chunk['content']}")
                if chunk["source"] not in sources:
                    sources.append(chunk["source"])
            context = "\n".join(context_lines)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
                {"role": "user", "content": query},
            ]

            answer_parts = []
            for chunk in self.chat_client.complete_streaming_chat(messages):
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    answer_parts.append(content)
            answer = "".join(answer_parts)
            answer += f"\n\n(Kaynaklar: {', '.join(sources)})"

            self.root.after(0, lambda: self._append_chat("Asistan", answer))
        except Exception as e:
            self.root.after(0, lambda: self._append_chat("Hata", str(e)))
        finally:
            self.root.after(0, self._reset_input)

    def _reset_input(self):
        self.entry.config(state="normal")
        self.send_button.config(state="normal")
        self.entry.focus()
        self.status_label.config(
            text=f"Hazir! Veritabaninda {len(self.all_chunks)} bilgi parcasi var.",
            fg="#4caf50",
        )


def main():
    root = tk.Tk()
    app = RagApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
