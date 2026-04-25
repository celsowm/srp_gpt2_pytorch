import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import glob
from pathlib import Path
from srp_gpt2.data.bpe import SimpleSentencePieceBPE

class TokenizerXrayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SRP GPT-2: BPE Tokenizer X-Ray")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")

        self.bpe = SimpleSentencePieceBPE()
        self.is_training = False
        
        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Treeview", background="#252526", foreground="#ffffff", fieldbackground="#252526", borderwidth=0)
        style.map("Treeview", background=[('selected', '#37373d')])

    def _build_ui(self):
        # Main Layout
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main_frame, text="BPE Tokenizer Training Visualizer", style="Header.TLabel")
        header.pack(pady=(0, 20))

        # Config Panel
        config_frame = ttk.LabelFrame(main_frame, text=" Configuration ", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 20))

        # Input Row
        input_row = ttk.Frame(config_frame)
        input_row.pack(fill=tk.X, pady=5)
        ttk.Label(input_row, text="Input Path:", width=12).pack(side=tk.LEFT)
        self.input_path = tk.StringVar(value="dataset_livros_ptbr/*.txt")
        ttk.Entry(input_row, textvariable=self.input_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_row, text="Browse", command=self._browse_input).pack(side=tk.LEFT)

        # Vocab Row
        vocab_row = ttk.Frame(config_frame)
        vocab_row.pack(fill=tk.X, pady=5)
        ttk.Label(vocab_row, text="Vocab Size:", width=12).pack(side=tk.LEFT)
        self.vocab_size = tk.IntVar(value=1000)
        ttk.Scale(vocab_row, from_=100, to=32000, variable=self.vocab_size, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(vocab_row, textvariable=self.vocab_size, width=6).pack(side=tk.LEFT)

        # Action Row
        action_row = ttk.Frame(config_frame)
        action_row.pack(fill=tk.X, pady=10)
        self.btn_train = ttk.Button(action_row, text="Start Training", command=self._start_training)
        self.btn_train.pack(side=tk.LEFT, padx=5)
        
        # Stats Display
        self.stats_label = ttk.Label(main_frame, text="Ready to train...")
        self.stats_label.pack(fill=tk.X, pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)

        # Dashboard (Split View)
        dashboard = ttk.Frame(main_frame)
        dashboard.pack(fill=tk.BOTH, expand=True)

        # Left: Merges Tree
        tree_frame = ttk.LabelFrame(dashboard, text=" Learned Merges ", padding="5")
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.tree = ttk.Treeview(tree_frame, columns=("Rank", "Left", "Right", "Merged", "Freq"), show='headings')
        self.tree.heading("Rank", text="#")
        self.tree.heading("Left", text="Left")
        self.tree.heading("Right", text="Right")
        self.tree.heading("Merged", text="Merged")
        self.tree.heading("Freq", text="Freq")
        self.tree.column("Rank", width=50)
        self.tree.column("Freq", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Right: Visualization Area
        viz_frame = ttk.LabelFrame(dashboard, text=" Live Merge View ", padding="5")
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(viz_frame, bg="#252526", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _browse_input(self):
        path = filedialog.askdirectory()
        if path:
            self.input_path.set(f"{path}/*.txt")

    def _start_training(self):
        if self.is_training: return
        
        pattern = self.input_path.get()
        files = glob.glob(pattern)
        if not files:
            messagebox.showerror("Error", f"No files found for: {pattern}")
            return

        self.is_training = True
        self.btn_train.config(state=tk.DISABLED)
        self.tree.delete(*self.tree.get_children())
        self.progress['value'] = 0
        self.progress['maximum'] = self.vocab_size.get()

        # Start training in a separate thread
        thread = threading.Thread(target=self._run_train, daemon=True)
        thread.start()

    def _run_train(self):
        try:
            pattern = self.input_path.get()
            files = glob.glob(pattern)
            
            def iter_files():
                for f in files:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                        for line in file: yield line

            self.bpe.train(
                source=iter_files(),
                vocab_size=self.vocab_size.get(),
                model_prefix="data/tokenizer/xray_model",
                verbose=False,
                progress_callback=self._on_progress
            )
            self.root.after(0, lambda: messagebox.showinfo("Success", "Training Complete!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.is_training = False
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))

    def _on_progress(self, data):
        # This is called from the training thread
        self.root.after(0, lambda: self._update_ui(data))

    def _update_ui(self, data):
        vocab_size = data['vocab_size']
        merges = data['merges']
        left, right, merged = data['last_merge']
        freq = data['freq']

        # Update labels and progress
        self.stats_label.config(text=f"Vocab: {vocab_size} | Merges: {merges} | Freq: {freq}")
        self.progress['value'] = vocab_size

        # Update Treeview (show last 1000)
        if merges % 5 == 0: # Throttle UI updates for performance
            item = self.tree.insert("", 0, values=(merges, left, right, merged, freq))
            # Keep tree small for UI responsiveness
            if len(self.tree.get_children()) > 100:
                self.tree.delete(self.tree.get_children()[-1])

        # Update Visualization Canvas
        self._draw_merge_viz(left, right, merged, freq)

    def _draw_merge_viz(self, left, right, merged, freq):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # Draw a "merging" animation effect
        cx, cy = w/2, h/2
        
        # Blocks
        self.canvas.create_rectangle(cx-120, cy-30, cx-20, cy+30, fill="#007acc", outline="#ffffff")
        self.canvas.create_text(cx-70, cy, text=left, fill="white", font=("Segoe UI", 12, "bold"))
        
        self.canvas.create_text(cx, cy, text="+", fill="white", font=("Segoe UI", 16))
        
        self.canvas.create_rectangle(cx+20, cy-30, cx+120, cy+30, fill="#007acc", outline="#ffffff")
        self.canvas.create_text(cx+70, cy, text=right, fill="white", font=("Segoe UI", 12, "bold"))

        self.canvas.create_text(cx, cy+60, text="↓↓", fill="#4ec9b0", font=("Segoe UI", 16))
        
        self.canvas.create_rectangle(cx-50, cy+90, cx+50, cy+150, fill="#4ec9b0", outline="#ffffff")
        self.canvas.create_text(cx, cy+120, text=merged, fill="black", font=("Segoe UI", 14, "bold"))
        
        self.canvas.create_text(cx, cy+180, text=f"Pair Frequency: {freq:,}", fill="#cccccc")

if __name__ == "__main__":
    root = tk.Tk()
    app = TokenizerXrayApp(root)
    root.mainloop()
