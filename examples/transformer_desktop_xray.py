"""Interactive desktop xray for tiny Transformer training and generation."""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from srp_gpt2.xray import (
    LiveGenerationStep,
    LiveTrainingStep,
    TensorSummary,
    TinyLiveGenerationSession,
    TinyLiveTrainingSession,
    TransformerTrace,
    resolve_xray_device,
    xray_tokenizer_label,
)


STAGES = ["Tokens", "Embeddings", "Attention", "MLP", "LayerNorm", "Logits", "Next token"]
STAGE_DESCRIPTIONS = {
    "Tokens": "Texto vira IDs numericos. O modelo nunca recebe texto cru.",
    "Embeddings": "Cada ID vira um vetor denso que carrega significado e posicao.",
    "Attention": "Cada token consulta tokens anteriores usando a mascara causal.",
    "MLP": "A rede feed-forward mistura features em cada posicao.",
    "LayerNorm": "Normaliza as ativacoes para estabilizar o fluxo.",
    "Logits": "Pontuacoes para todos os proximos tokens possiveis.",
    "Next token": "A distribuicao vira a proxima escolha do modelo.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop Transformer xray.")
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--text-file", type=Path, default=Path("data/tiny.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny_xray/last.pt"))
    parser.add_argument("--prompt", type=str, default="O rato")
    parser.add_argument("--strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--tokenizer", choices=["gpt2", "byte-debug"], default="gpt2")
    args = parser.parse_args()

    app = TransformerDesktopXray(args)
    app.mainloop()


class TransformerDesktopXray(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.title("SRP GPT-2 Transformer Xray")
        self.geometry("1360x840")
        self.minsize(1120, 720)
        self.args = args
        self.device = resolve_xray_device(args.device)
        self.result_queue: queue.Queue[object] = queue.Queue()
        self.busy = False
        self.playing = False
        self.current_trace: TransformerTrace | None = None
        self.current_stage = 0
        self.train_session: TinyLiveTrainingSession | None = None
        self.generate_session: TinyLiveGenerationSession | None = None

        self._build_widgets()
        self.mode_var.set(args.mode)
        self.strategy_var.set(args.strategy)
        self.tokenizer_var.set(args.tokenizer)
        self.prompt_var.set(args.prompt)
        self._set_status(f"device={self.device}; carregando sessao...")
        self.after(50, self._initialize_session)
        self.after(50, self._poll_results)

    def _build_widgets(self) -> None:
        self.configure(background="#eef2f7")
        style = ttk.Style(self)
        style.configure("TFrame", background="#eef2f7")
        style.configure("Toolbar.TFrame", background="#ffffff")
        style.configure("TLabel", background="#eef2f7", font=("Segoe UI", 9))
        style.configure("Toolbar.TLabel", background="#ffffff", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 4))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=10, style="Toolbar.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(10, weight=1)

        self.mode_var = tk.StringVar()
        ttk.Label(toolbar, text="Modo", style="Toolbar.TLabel").grid(row=0, column=0, padx=(0, 4))
        mode = ttk.Combobox(
            toolbar,
            textvariable=self.mode_var,
            values=["train", "generate"],
            width=10,
            state="readonly",
        )
        mode.grid(row=0, column=1, padx=(0, 10))
        mode.bind("<<ComboboxSelected>>", lambda _event: self._on_mode_change())

        self.step_button = ttk.Button(toolbar, text="1 step", command=self.run_one_step)
        self.step_button.grid(row=0, column=2, padx=4)
        self.play_button = ttk.Button(toolbar, text="play", command=self.toggle_play)
        self.play_button.grid(row=0, column=3, padx=4)
        ttk.Button(toolbar, text="reset", command=self.reset_session).grid(row=0, column=4, padx=4)

        ttk.Label(toolbar, text="Velocidade", style="Toolbar.TLabel").grid(row=0, column=5, padx=(14, 4))
        self.speed_var = tk.IntVar(value=450)
        ttk.Scale(
            toolbar,
            from_=1200,
            to=80,
            variable=self.speed_var,
            orient="horizontal",
            length=170,
        ).grid(row=0, column=6, padx=(0, 10))

        self.strategy_var = tk.StringVar()
        ttk.Label(toolbar, text="Decoding", style="Toolbar.TLabel").grid(row=0, column=7, padx=(0, 4))
        ttk.Combobox(
            toolbar,
            textvariable=self.strategy_var,
            values=["greedy", "sample"],
            width=9,
            state="readonly",
        ).grid(row=0, column=8, sticky="w")

        ttk.Label(toolbar, text="Tokenizer", style="Toolbar.TLabel").grid(row=0, column=9, padx=(14, 4))
        self.tokenizer_var = tk.StringVar()
        tokenizer_combo = ttk.Combobox(
            toolbar,
            textvariable=self.tokenizer_var,
            values=["gpt2", "byte-debug"],
            width=12,
            state="readonly",
        )
        tokenizer_combo.grid(row=0, column=10, sticky="w")
        tokenizer_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_tokenizer_change())

        ttk.Label(toolbar, text="Prompt", style="Toolbar.TLabel").grid(row=1, column=0, padx=(0, 4), pady=(8, 0))
        self.prompt_var = tk.StringVar()
        ttk.Entry(toolbar, textvariable=self.prompt_var).grid(
            row=1,
            column=1,
            columnspan=5,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(toolbar, text="aplicar prompt", command=self.reset_generation_prompt).grid(
            row=1,
            column=6,
            padx=4,
            pady=(8, 0),
        )
        self.status_var = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.status_var, style="Toolbar.TLabel").grid(
            row=1,
            column=7,
            columnspan=4,
            sticky="e",
            pady=(8, 0),
        )

        main = ttk.PanedWindow(self, orient="horizontal")
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = ttk.Frame(main, padding=0)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, background="#f8fafc", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        main.add(left, weight=4)

        right = ttk.Frame(main, padding=(10, 0, 0, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="Inspecao", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.metrics = tk.Text(
            right,
            width=42,
            height=30,
            wrap="word",
            borderwidth=0,
            background="#ffffff",
            foreground="#111827",
            font=("Consolas", 10),
            padx=12,
            pady=10,
        )
        self.metrics.grid(row=1, column=0, sticky="nsew")
        main.add(right, weight=1)

    def _initialize_session(self) -> None:
        try:
            if self.mode_var.get() == "train":
                self._ensure_train_session()
            else:
                self._ensure_generate_session()
            self._set_status(f"device={self.device}; pronto")
            self.run_one_step()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._show_error(exc)

    def _ensure_train_session(self) -> TinyLiveTrainingSession:
        if self.train_session is None:
            self.train_session = TinyLiveTrainingSession(
                text_file=self.args.text_file,
                device=self.device,
                tokenizer_name=self.tokenizer_var.get(),
            )
        return self.train_session

    def _ensure_generate_session(self) -> TinyLiveGenerationSession:
        if self.generate_session is None:
            if not self.args.checkpoint.exists():
                raise FileNotFoundError(
                    f"checkpoint nao encontrado: {self.args.checkpoint}. "
                    "Treine antes com examples/train_tiny_xray.py --mode overfit."
                )
            self.generate_session = TinyLiveGenerationSession(
                checkpoint=self.args.checkpoint,
                prompt=self.prompt_var.get(),
                device=self.device,
                strategy=self.strategy_var.get(),
                tokenizer_name=self.tokenizer_var.get(),
            )
        else:
            self.generate_session.strategy = self.strategy_var.get()
        return self.generate_session

    def _on_mode_change(self) -> None:
        self.playing = False
        self.play_button.configure(text="play")
        self.run_one_step()

    def _on_tokenizer_change(self) -> None:
        self.playing = False
        self.play_button.configure(text="play")
        self.train_session = None
        self.generate_session = None
        self._clear_display(f"tokenizer alterado para {xray_tokenizer_label(self.tokenizer_var.get())}")
        self.run_one_step()

    def reset_generation_prompt(self) -> None:
        if self.generate_session is not None:
            self.generate_session.strategy = self.strategy_var.get()
            self.generate_session.reset(self.prompt_var.get())
        self.mode_var.set("generate")
        self.run_one_step()

    def reset_session(self) -> None:
        self.playing = False
        self.play_button.configure(text="play")
        try:
            if self.mode_var.get() == "train":
                self._ensure_train_session().reset()
            else:
                if self.generate_session is not None:
                    self.generate_session.reset(self.prompt_var.get())
            self._clear_display("sessao reiniciada")
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._show_error(exc)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="pause" if self.playing else "play")
        if self.playing and not self.busy:
            self.run_one_step()

    def run_one_step(self) -> None:
        if self.busy:
            return
        self.busy = True
        self._set_status(f"device={self.device}; calculando...")
        thread = threading.Thread(target=self._worker_step, daemon=True)
        thread.start()

    def _worker_step(self) -> None:
        try:
            if self.mode_var.get() == "train":
                result = self._ensure_train_session().step()
            else:
                session = self._ensure_generate_session()
                session.strategy = self.strategy_var.get()
                result = session.step()
            self.result_queue.put(result)
        except Exception as exc:  # noqa: BLE001 - sent to UI thread
            self.result_queue.put(exc)

    def _poll_results(self) -> None:
        try:
            while True:
                item = self.result_queue.get_nowait()
                self.busy = False
                if isinstance(item, Exception):
                    self.playing = False
                    self.play_button.configure(text="play")
                    self._show_error(item)
                else:
                    self._handle_step_result(item)
        except queue.Empty:
            pass
        self.after(50, self._poll_results)

    def _handle_step_result(self, item: object) -> None:
        if isinstance(item, LiveTrainingStep):
            self.current_trace = item.trace_after
            self._update_metrics_for_training(item)
        elif isinstance(item, LiveGenerationStep):
            self.current_trace = item.trace
            self._update_metrics_for_generation(item)
        else:
            return
        self.current_stage = 0
        self._animate_current_trace()

    def _animate_current_trace(self) -> None:
        if self.current_trace is None:
            return
        self._draw_trace(self.current_trace, self.current_stage)
        if self.current_stage < len(STAGES) - 1:
            self.current_stage += 1
            self.after(int(self.speed_var.get()), self._animate_current_trace)
            return
        self._set_status(f"device={self.device}; pronto")
        if self.playing:
            self.after(int(self.speed_var.get()), self.run_one_step)

    def _draw_trace(self, trace: TransformerTrace, active_stage: int) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 900)
        height = max(canvas.winfo_height(), 620)
        canvas.create_rectangle(0, 0, width, height, fill="#f8fafc", outline="")
        canvas.create_text(
            32,
            28,
            anchor="w",
            text="Token atravessando o Transformer",
            fill="#1f2937",
            font=("Segoe UI", 18, "bold"),
        )
        stage = STAGES[active_stage]
        canvas.create_text(
            34,
            58,
            anchor="w",
            text=(
                f"Tokenizer: {xray_tokenizer_label(self.tokenizer_var.get())}. "
                "Tokens sao pedacos aprendidos pelo tokenizer, nao letras. "
                f"Etapa atual: {stage} - {STAGE_DESCRIPTIONS[stage]}"
            ),
            fill="#475569",
            font=("Segoe UI", 11),
        )
        self._draw_tokens(canvas, trace, 34, 92, width - 70)
        self._draw_pipeline(canvas, trace, active_stage, width, 205)
        graph_right = max(520, width - 430)
        self._draw_neural_graph(canvas, active_stage, 70, 410, graph_right)
        self._draw_attention(canvas, trace, width - 360, 405, 310, 250)

    def _draw_tokens(
        self,
        canvas: tk.Canvas,
        trace: TransformerTrace,
        x: int,
        y: int,
        max_width: int,
    ) -> None:
        tokens = trace.input_tokens[-18:]
        canvas.create_text(
            x,
            y - 18,
            anchor="w",
            text="Contexto visivel: token text em cima, token_id embaixo",
            fill="#334155",
            font=("Segoe UI", 11, "bold"),
        )
        cursor = x
        for row in tokens:
            label = row["text"]
            box_width = max(48, min(112, 26 + len(label) * 9))
            if cursor + box_width > max_width:
                break
            canvas.create_rectangle(cursor, y, cursor + box_width, y + 42, fill="#ffffff", outline="#cbd5e1", width=1)
            canvas.create_text(cursor + box_width / 2, y + 13, text=label, font=("Consolas", 10, "bold"), fill="#0f172a")
            canvas.create_text(
                cursor + box_width / 2,
                y + 30,
                text=str(row["token_id"]),
                fill="#64748b",
                font=("Consolas", 8),
            )
            cursor += box_width + 6

    def _draw_pipeline(
        self,
        canvas: tk.Canvas,
        trace: TransformerTrace,
        active_stage: int,
        width: int,
        y: int,
    ) -> None:
        summaries = [
            trace.embeddings,
            trace.embeddings,
            trace.blocks[-1].attention if trace.blocks else trace.embeddings,
            trace.blocks[-1].mlp if trace.blocks else trace.embeddings,
            trace.final_norm,
            trace.logits,
            trace.logits,
        ]
        left = 64
        usable = width - 150
        gap = usable / (len(STAGES) - 1)
        points = [(left + idx * gap, y) for idx in range(len(STAGES))]
        for idx in range(len(points) - 1):
            canvas.create_line(*points[idx], *points[idx + 1], fill="#94a3b8", width=4, arrow=tk.LAST)
        for idx, ((px, py), stage) in enumerate(zip(points, STAGES, strict=True)):
            active = idx == active_stage
            fill = "#2563eb" if active else "#ffffff"
            outline = "#1d4ed8" if active else "#cbd5e1"
            text_fill = "#ffffff" if active else "#111827"
            canvas.create_rectangle(px - 54, py - 36, px + 54, py + 36, fill=fill, outline=outline, width=2)
            canvas.create_text(px, py - 10, text=stage, fill=text_fill, width=100, font=("Segoe UI", 10, "bold"))
            if active:
                canvas.create_text(px, py + 15, text="agora", fill="#dbeafe", font=("Segoe UI", 8, "bold"))
            canvas.create_text(
                px,
                py + 72,
                text=self._summary_label(summaries[idx]),
                fill="#475569",
                width=135,
                font=("Consolas", 9),
            )
        top = trace.next_token.top_tokens[:3]
        top_text = " | ".join(f"{token.text}:{token.probability:.2f}" for token in top)
        canvas.create_rectangle(width - 360, y + 88, width - 44, y + 134, fill="#ffffff", outline="#cbd5e1")
        canvas.create_text(
            width - 346,
            y + 111,
            anchor="w",
            text=f"Top tokens: {top_text}",
            fill="#0f172a",
            font=("Segoe UI", 10, "bold"),
        )

    def _draw_neural_graph(
        self,
        canvas: tk.Canvas,
        active_stage: int,
        left: int,
        top: int,
        right: int,
    ) -> None:
        canvas.create_text(left, top - 34, anchor="w", text="Grafo neural simplificado", fill="#0f172a", font=("Segoe UI", 13, "bold"))
        canvas.create_text(
            left,
            top - 14,
            anchor="w",
            text="Uma analogia visual: sinais fluem por camadas, mas as metricas acima vem do Transformer real.",
            fill="#64748b",
            font=("Segoe UI", 9),
        )
        layers = 5
        nodes = 5
        x_gap = (right - left) / max(1, layers - 1)
        positions: list[list[tuple[float, float]]] = []
        for layer in range(layers):
            layer_nodes = []
            for node in range(nodes):
                px = left + layer * x_gap
                py = top + node * 34
                layer_nodes.append((px, py))
            positions.append(layer_nodes)
        active_layer = min(layers - 1, max(0, int(active_stage * layers / len(STAGES))))
        for layer in range(layers - 1):
            color = "#60a5fa" if layer == active_layer else "#cbd5e1"
            for start in positions[layer]:
                for end in positions[layer + 1]:
                    canvas.create_line(*start, *end, fill=color)
        for layer_idx, layer in enumerate(positions):
            for node_idx, (px, py) in enumerate(layer):
                active = layer_idx == active_layer and (node_idx + active_stage) % 2 == 0
                canvas.create_oval(
                    px - 10,
                    py - 10,
                    px + 10,
                    py + 10,
                    fill="#f97316" if active else "#ffffff",
                    outline="#475569",
                    width=2 if active else 1,
                )

    def _draw_attention(
        self,
        canvas: tk.Canvas,
        trace: TransformerTrace,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        canvas.create_text(x, y - 34, anchor="w", text="Atencao causal", fill="#0f172a", font=("Segoe UI", 13, "bold"))
        canvas.create_text(
            x,
            y - 14,
            anchor="w",
            text="Mais escuro = mais peso. Branco acima da diagonal = futuro bloqueado.",
            fill="#64748b",
            font=("Segoe UI", 9),
            width=width,
        )
        if not trace.blocks:
            return
        attention = trace.blocks[-1].attention_map
        size = len(attention)
        if size == 0:
            return
        cell = min(width / size, height / size)
        canvas.create_rectangle(x - 1, y - 1, x + cell * size + 1, y + cell * size + 1, fill="#ffffff", outline="#cbd5e1")
        for row_idx, row in enumerate(attention):
            for col_idx, value in enumerate(row):
                strength = min(1.0, value * 5.0)
                red = int(239 - 202 * strength)
                green = int(246 - 147 * strength)
                blue = int(255 - 20 * strength)
                color = f"#{red:02x}{green:02x}{blue:02x}"
                x0 = x + col_idx * cell
                y0 = y + row_idx * cell
                canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=color, outline="")

    def _update_metrics_for_training(self, item: LiveTrainingStep) -> None:
        active_top = item.trace_after.next_token.top_tokens[0]
        lines = [
            "TREINO AO VIVO",
            "",
            "O que aconteceu neste step:",
            "- Um minibatch passou pelo Transformer.",
            "- A loss mediu o erro de prever o proximo token.",
            "- O backprop ajustou os pesos.",
            "",
            f"step              {item.step}",
            f"loss              {item.loss:.4f}",
            f"perplexity        {item.perplexity:.2f}",
            f"learning rate     {item.learning_rate:.2e}",
            f"grad norm         {item.grad_norm:.3f}",
            f"param norm        {item.param_norm:.3f}",
            "",
            f"Mais provavel agora: {active_top.text!r}  p={active_top.probability:.3f}",
            "",
            "Ranking do proximo token:",
            self._top_tokens_text(item.trace_after),
        ]
        self._set_metrics("\n".join(lines))

    def _update_metrics_for_generation(self, item: LiveGenerationStep) -> None:
        confidence = item.trace.next_token.confidence
        entropy = item.trace.next_token.entropy
        lines = [
            "INFERENCIA AO VIVO",
            "",
            "O que aconteceu neste token:",
            "- O contexto foi convertido em vetores.",
            "- A atencao consultou apenas o passado.",
            "- Os logits viraram uma distribuicao.",
            "- O decoder escolheu o proximo token.",
            "",
            f"step              {item.step}",
            f"token escolhido   {item.chosen_text!r} ({item.chosen_id})",
            f"confidence        {confidence:.3f}",
            f"entropy           {entropy:.3f}",
            "",
            "Ranking antes da escolha:",
            self._top_tokens_text(item.trace),
            "",
            "Texto acumulado:",
            item.accumulated_text,
        ]
        self._set_metrics("\n".join(lines))

    def _top_tokens_text(self, trace: TransformerTrace) -> str:
        return "\n".join(
            f"{idx}. {token.text!r:<8} id={token.token_id:<4} p={token.probability:.3f}"
            for idx, token in enumerate(trace.next_token.top_tokens, start=1)
        )

    def _summary_label(self, summary: TensorSummary) -> str:
        return f"shape={summary.shape}\nnorm={summary.norm:.2f}\nmean={summary.mean:.2f}"

    def _set_metrics(self, text: str) -> None:
        self.metrics.configure(state="normal")
        self.metrics.delete("1.0", tk.END)
        self.metrics.insert(tk.END, text)
        self.metrics.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _clear_display(self, text: str) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(40, 40, anchor="w", text=text, font=("Segoe UI", 14, "bold"))
        self._set_metrics(text)
        self._set_status(f"device={self.device}; pronto")

    def _show_error(self, exc: Exception) -> None:
        self.busy = False
        self._set_status(f"device={self.device}; erro")
        messagebox.showerror("Transformer Xray", str(exc))


if __name__ == "__main__":
    main()
