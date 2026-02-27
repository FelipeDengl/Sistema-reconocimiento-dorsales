#!/usr/bin/env python3
import sys, os, csv, json, shutil, tempfile, textwrap, subprocess, re, threading, queue
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QWheelEvent, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QMessageBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QListWidget, QListWidgetItem, QProgressBar, QCheckBox
)

# -------------------- Configuración OCR --------------------
MIN_CONF = 0.50
MIN_HEIGHT_RATIO = 0.05
MIN_DIGITS = 2
MAX_DIGITS = 4
MARGIN_RATIO = 0.03
MAX_LONG_SIDE = 1600
BATCH_SIZE = 16
AUTO_GPU = True

@dataclass
class Detection:
    box: List[List[int]]
    text: str
    conf: float

@dataclass
class ImageEntry:
    path: Path
    detections: List[Detection] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)

def readable_time():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def ensure_out_dirs(base: Path) -> Path:
    out = base / "out"
    out.mkdir(exist_ok=True)
    (out / "discard").mkdir(exist_ok=True)
    return out

def append_csv_row(csv_path: Path, row: list, header: list):
    existed = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not existed:
            w.writerow(header)
        w.writerow(row)

def clean_number(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def parse_numbers_free(text: str) -> List[str]:
    parts = re.split(r"[,\s;]+", text.strip())
    out, seen = [], set()
    for p in parts:
        num = clean_number(p)
        if num and num not in seen:
            out.append(num); seen.add(num)
    return out

# -------------------- Visor --------------------
class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.setBackgroundBrush(QColor(30, 30, 30))
        self._zoom = 0
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def set_theme(self, dark: bool):
        # True = oscuro, False = claro
        self.setBackgroundBrush(QColor(30, 30, 30) if dark else QColor(255, 255, 255))

    def set_pixmap(self, pix: QPixmap):
        self.scene().clear()
        self.pixmap_item = QGraphicsPixmapItem(pix)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fit()

    def fit(self):
        if self.pixmap_item:
            self._zoom = 0
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        if not self.pixmap_item:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)
        self._zoom += 1 if factor > 1 else -1

    def mouseDoubleClickEvent(self, e):
        self.fit()
        super().mouseDoubleClickEvent(e)

# -------------------- App --------------------
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real Vision OCR")
        self.resize(1400, 900)

        self.base_folder: Optional[Path] = None
        self.out_dir: Optional[Path] = None
        self.csv_path: Optional[Path] = None

        self.entries: List[ImageEntry] = []
        self.img_idx = 0
        self.cand_idx = -1
        self.accepted_paths: Dict[str, Path] = {}

        self.dark_mode = True  # tema inicial

        self._build_ui()

    def _big(self, w):
        w.setMinimumHeight(44)
        f = w.font(); f.setPointSize(12); w.setFont(f)

    def _label_big(self, w: QLabel):
        f = w.font(); f.setPointSize(11); w.setFont(f)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()

        # Barra superior con botón de tema
        topbar = QHBoxLayout()
        topbar.addStretch()
        self.btn_theme = QPushButton("🌓")
        self.btn_theme.setFixedSize(28, 28)
        self.btn_theme.setToolTip("Cambiar tema (Ctrl+T)")
        self.btn_theme.setStyleSheet("QPushButton{font-size:14px; padding:0;}")
        self.btn_theme.clicked.connect(self.toggle_theme)
        topbar.addWidget(self.btn_theme)
        left.addLayout(topbar)

        self.view = ImageView()
        self.view.set_theme(self.dark_mode)
        left.addWidget(self.view)

        self.info_label = QLabel("Archivo: - (0/0)   Número actual: -")
        self._label_big(self.info_label)
        left.addWidget(self.info_label)
        root.addLayout(left, 3)

        right = QVBoxLayout()

        self.chk_use_ocr = QCheckBox("Utilizar detección de números (OCR)")
        self.chk_use_ocr.setChecked(True)
        self._label_big(self.chk_use_ocr)
        right.addWidget(self.chk_use_ocr)

        self.btn_open = QPushButton("Abrir carpeta")
        self._big(self.btn_open)
        self.btn_open.clicked.connect(self.open_and_process)
        right.addWidget(self.btn_open)

        self.folder_label = QLabel("Carpeta: -"); self.folder_label.setWordWrap(True)
        self._label_big(self.folder_label)
        right.addWidget(self.folder_label)

        self.images_progress_label = QLabel("Imágenes: 0/0")
        self._label_big(self.images_progress_label)
        right.addWidget(self.images_progress_label)

        lab_num = QLabel("Número actual (editable):")
        self._label_big(lab_num)
        right.addWidget(lab_num)
        self.current_num_edit = QLineEdit()
        self.current_num_edit.setPlaceholderText("-")
        self._big(self.current_num_edit)
        right.addWidget(self.current_num_edit)

        row = QHBoxLayout()
        self.btn_accept = QPushButton("Aceptar (Enter)")
        self._big(self.btn_accept)
        self.btn_reject = QPushButton("Rechazar (Supr)")
        self._big(self.btn_reject)
        self.btn_accept.clicked.connect(self.accept_current)
        self.btn_reject.clicked.connect(self.reject_current)
        row.addWidget(self.btn_accept); row.addWidget(self.btn_reject)
        right.addLayout(row)

        self.btn_skip = QPushButton("Saltar imagen (N)")
        self._big(self.btn_skip)
        self.btn_skip.clicked.connect(self.skip_image)
        right.addWidget(self.btn_skip)

        right.addSpacing(8)
        lab_acc = QLabel("Aceptados en esta imagen:")
        self._label_big(lab_acc)
        right.addWidget(lab_acc)
        self.accepted_list = QListWidget()
        self.accepted_list.setFixedHeight(200)
        self.accepted_list.setStyleSheet("""
            QListWidget::item:hover { background: #b30000; color: white; }
            QListWidget { font-size: 12pt; }
        """)
        self.accepted_list.itemClicked.connect(self._on_item_clicked_delete)
        right.addWidget(self.accepted_list)

        self.btn_remove_selected = QPushButton("Eliminar seleccionado de la lista (y deshacer copia)")
        self._big(self.btn_remove_selected)
        self.btn_remove_selected.clicked.connect(self.remove_selected_accepted)
        right.addWidget(self.btn_remove_selected)

        right.addSpacing(12)
        lab_missing = QLabel("Faltantes (opcional):")
        self._label_big(lab_missing)
        right.addWidget(lab_missing)
        self.missing_edit = QLineEdit()
        self.missing_edit.setPlaceholderText("Ej: 102 103  o  102,103")
        self._big(self.missing_edit)
        right.addWidget(self.missing_edit)
        self.btn_confirm_missing = QPushButton("Confirmar faltantes (Enter)")
        self._big(self.btn_confirm_missing)
        self.btn_confirm_missing.clicked.connect(self.confirm_missing)
        right.addWidget(self.btn_confirm_missing)

        right.addSpacing(12)
        self.progress_label = QLabel("OCR: esperando…")
        self._label_big(self.progress_label)
        right.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { height: 28px; font-size: 12pt; }")
        right.addWidget(self.progress_bar)

        right.addStretch()
        self.status_label = QLabel("Estado: listo")
        self._label_big(self.status_label)
        right.addWidget(self.status_label)

        root.addLayout(right, 1)

        self.current_num_edit.installEventFilter(self)
        self.missing_edit.installEventFilter(self)

        # Aplica tema global inicial (oscuro por defecto)
        self._apply_theme()

    # ---------- Tema (oscuro/claro) ----------
    def _apply_theme(self):
        app = QApplication.instance()
        if self.dark_mode:
            app.setStyleSheet("""
                QMainWindow, QWidget { background: #1e1f22; color: #e6e6e6; }
                QLabel { color: #e6e6e6; }
                QLineEdit, QListWidget, QProgressBar, QCheckBox, QGraphicsView {
                    background: #2a2b2f; color: #e6e6e6; border: 1px solid #3a3b40;
                }
                QLineEdit:focus, QListWidget:focus { border: 1px solid #6a6cff; }
                QPushButton {
                    background: #2f3136; color: #e6e6e6; border: 1px solid #3a3b40; padding: 6px 10px;
                }
                QPushButton:hover { background: #3a3c41; }
                QPushButton:disabled { color: #888; border-color: #2c2d31; }
                QProgressBar { text-align: center; }
                QProgressBar::chunk { background: #4b9fff; }
                QListWidget::item:hover { background: #b30000; color: white; }
                QScrollBar:vertical { background:#2a2b2f; width:10px; margin:0; }
                QScrollBar::handle:vertical { background:#44464c; min-height:20px; border-radius:4px; }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
            """)
            self.view.set_theme(True)
        else:
            app.setStyleSheet("""
                QMainWindow, QWidget { background: #ffffff; color: #111111; }
                QLabel { color: #111111; }
                QLineEdit, QListWidget, QProgressBar, QCheckBox, QGraphicsView {
                    background: #ffffff; color: #111111; border: 1px solid #c9c9c9;
                }
                QLineEdit:focus, QListWidget:focus { border: 1px solid #4b79ff; }
                QPushButton {
                    background: #f2f2f2; color: #111111; border: 1px solid #c9c9c9; padding: 6px 10px;
                }
                QPushButton:hover { background: #e8e8e8; }
                QPushButton:disabled { color: #9a9a9a; border-color: #dcdcdc; }
                QProgressBar { text-align: center; }
                QProgressBar::chunk { background: #4b79ff; }
                QListWidget::item:hover { background: #ffeded; color: #b30000; }
                QScrollBar:vertical { background:#ffffff; width:10px; margin:0; }
                QScrollBar::handle:vertical { background:#cfcfcf; min-height:20px; border-radius:4px; }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
            """)
            self.view.set_theme(False)

    # ---------- OCR cajas reescaladas a original----------
    def run_ocr_in_subprocess(self, image_paths: List[Path]) -> List[ImageEntry]:
        tmpdir = Path(tempfile.mkdtemp(prefix="ocr_batch_"))
        inp_json = tmpdir / "in_images.json"
        out_json = tmpdir / "out_results.json"
        inp_json.write_text(json.dumps([str(p) for p in image_paths]), encoding="utf-8")

        script = textwrap.dedent(f"""
        import sys, os, json, ctypes, numpy as np
        from pathlib import Path

        MIN_CONF = {MIN_CONF}
        MIN_HEIGHT_RATIO = {MIN_HEIGHT_RATIO}
        MIN_DIGITS = {MIN_DIGITS}
        MAX_DIGITS = {MAX_DIGITS}
        MARGIN_RATIO = {MARGIN_RATIO}
        MAX_LONG_SIDE = {MAX_LONG_SIDE}
        BATCH_SIZE = {BATCH_SIZE}
        AUTO_GPU = {str(AUTO_GPU)}

        try:
            torch_lib = Path(sys.prefix) / 'Lib' / 'site-packages' / 'torch' / 'lib'
            if torch_lib.exists():
                os.environ['PATH'] = str(torch_lib) + os.pathsep + os.environ.get('PATH','')
                try: os.add_dll_directory(str(torch_lib))
                except Exception: pass
                for name in ['c10.dll','torch_cpu.dll','torch.dll','libiomp5md.dll','fbgemm.dll','sleef.dll','mkldnn.dll','torch_python.dll']:
                    p = torch_lib / name
                    if p.exists():
                        try: ctypes.WinDLL(str(p))
                        except Exception: pass
        except Exception:
            pass

        import easyocr, cv2, torch

        try:
            torch.set_num_threads(max(1, os.cpu_count() or 4))
            torch.set_num_interop_threads(1)
            os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4))))
            os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
        except Exception:
            pass

        use_gpu = False
        if AUTO_GPU:
            try:
                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False

        reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

        imgs = json.loads(Path(r'{str(inp_json)}').read_text(encoding='utf-8'))
        results = []
        total = len(imgs) or 1

        for i, s in enumerate(imgs, start=1):
            dets = []
            try:
                img = cv2.imread(s)
                h0, w0 = (img.shape[0], img.shape[1]) if img is not None else (1000, 1000)  # originales
                h, w = h0, w0
                scale = 1.0
                if max(h, w) > MAX_LONG_SIDE:
                    scale = MAX_LONG_SIDE / float(max(h, w))
                    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                    h, w = img.shape[:2]

                # umbrales en coordenadas ORIGINALES
                height_min = max(1, int(MIN_HEIGHT_RATIO * h0))
                mx = int(MARGIN_RATIO * w0); my = int(MARGIN_RATIO * h0)
                x_lo, x_hi = mx, w0 - mx
                y_lo, y_hi = my, h0 - my

                raw = reader.readtext(
                    img,
                    detail=1,
                    paragraph=False,
                    allowlist="0123456789",
                    decoder='greedy',
                    batch_size=BATCH_SIZE
                )

                for (box, text, conf) in raw:
                    # box en coords de imagen REDUCIDA; reescalar a ORIGINAL
                    b_res = [[float(x), float(y)] for (x,y) in box]
                    if scale != 1.0:
                        b = [[int(round(x/scale)), int(round(y/scale))] for (x,y) in b_res]
                    else:
                        b = [[int(round(x)), int(round(y))] for (x,y) in b_res]

                    n = ''.join(ch for ch in str(text) if ch.isdigit())
                    ys = [pt[1] for pt in b]; xs = [pt[0] for pt in b]
                    box_h = max(ys) - min(ys)
                    c = float(conf)

                    inside = all((x_lo <= x <= x_hi and y_lo <= y <= y_hi) for (x,y) in b)
                    if n and (MIN_DIGITS <= len(n) <= MAX_DIGITS) and c >= MIN_CONF and box_h >= height_min and inside:
                        dets.append({{'box': b, 'text': n, 'conf': c}})
            except Exception:
                dets = []

            results.append({{'path': s, 'detections': dets}})
            print(f"PROGRESS {{i}} {{total}} {{Path(s).name}}", flush=True)

        def np_to_py(o):
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.ndarray,)):  return o.tolist()
            return str(o)

        Path(r'{str(out_json)}').write_text(json.dumps(results, ensure_ascii=False, default=np_to_py), encoding='utf-8')
        print("DONE", flush=True)
        """)

        runner = tmpdir / "runner.py"
        runner.write_text(script, encoding="utf-8")

        creationflags = 0
        startupinfo = None
        if sys.platform == "win32":
            creationflags = 0x08000000
            import subprocess as _sp
            startupinfo = _sp.STARTUPINFO()
            startupinfo.dwFlags |= _sp.STARTF_USESHOWWINDOW

        proc = subprocess.Popen(
            [sys.executable, str(runner)],
            cwd=str(tmpdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
            startupinfo=startupinfo,
            bufsize=1
        )

        qlines: "queue.Queue[str]" = queue.Queue()

        def _pump(pipe, q):
            for line in iter(pipe.readline, ''):
                q.put(line.rstrip('\r\n'))
            pipe.close()

        t_out = threading.Thread(target=_pump, args=(proc.stdout, qlines), daemon=True)
        t_out.start()

        self.progress_label.setText("OCR: inicializando…")
        self.progress_bar.setRange(0, 0)

        total = len(image_paths) or 1
        seen_first = False

        while True:
            try:
                line = qlines.get(timeout=0.05)
            except queue.Empty:
                QApplication.processEvents()
                if proc.poll() is not None:
                    break
                continue

            # actualiza barra conforme llegan líneas
            if line.startswith("PROGRESS "):
                if not seen_first:
                    self.progress_bar.setRange(0, 100)
                    seen_first = True
                parts = line.split(" ", 3)
                try:
                    i = int(parts[1]); tot = int(parts[2]); fname = parts[3] if len(parts) > 3 else ""
                except Exception:
                    i, tot, fname = 1, total, ""
                percent = int(i * 100 / max(1, tot))
                self.progress_label.setText(f"OCR: {fname}  ({i}/{tot})")
                self.progress_bar.setValue(percent)
            elif line == "DONE":
                self.progress_bar.setValue(100)
                self.progress_label.setText("OCR: completado ✅")
                break
            else:
                self.progress_label.setText(f"OCR: {line}")

            QApplication.processEvents()

        stderr = proc.stderr.read()
        ret = proc.wait(timeout=5) if proc.poll() is None else proc.returncode
        if ret != 0:
            raise RuntimeError(f"OCR subproceso falló:\n{stderr}")

        raw = json.loads((tmpdir / "out_results.json").read_text(encoding="utf-8"))
        entries: List[ImageEntry] = []
        for item in raw:
            p = Path(item["path"])
            dets = [Detection(d["box"], d["text"], float(d["conf"])) for d in item.get("detections", [])]
            seen, cands = set(), []
            for d in dets:
                n = d.text
                if n and n not in seen:
                    seen.add(n); cands.append(n)
            entries.append(ImageEntry(path=p, detections=dets, candidates=cands))

        self.progress_bar.setValue(0)
        self.progress_label.setText("OCR: listo")
        self.progress_bar.setRange(0, 100)
        return entries

    # -------------------- Lógica UI --------------------
    def open_and_process(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta")
        if not folder:
            return
        self.base_folder = Path(folder)
        self.out_dir = ensure_out_dirs(self.base_folder)
        self.csv_path = self.base_folder / "actions.csv"
        self.folder_label.setText(str(self.base_folder))

        exts = {".jpg",".jpeg",".png",".bmp",".tiff",".webp",".JPG",".PNG",".JPEG"}
        image_paths = [p for p in sorted(self.base_folder.rglob("*")) if p.suffix in exts and p.is_file()]
        if not image_paths:
            QMessageBox.information(self, "Info", "No se encontraron imágenes.")
            return

        if not self.chk_use_ocr.isChecked():
            self.progress_label.setText("OCR: desactivado (modo manual)")
            self.progress_bar.setValue(0)
            self.progress_bar.setRange(0, 100)
            self.entries = [ImageEntry(path=p) for p in image_paths]
            self.img_idx = 0
            self.accepted_paths.clear()
            self.cand_idx = -1
            self.status_label.setText(f"Estado: revisión manual ({len(self.entries)} imágenes).")
            self.images_progress_label.setText(f"Imágenes: {self.img_idx+1}/{len(self.entries)}")
            self.show_current()
            return

        self.status_label.setText("Estado: procesando OCR…")
        try:
            self.entries = self.run_ocr_in_subprocess(image_paths)
        except Exception as e:
            QMessageBox.critical(self, "OCR", f"Error al procesar OCR:\n{e}")
            return

        self.img_idx = 0
        self.accepted_paths.clear()
        self.cand_idx = 0 if self.entries and self.entries[0].candidates else -1
        self.status_label.setText(f"Estado: revisión ({len(self.entries)} imágenes).")
        self.images_progress_label.setText(f"Imágenes: {self.img_idx+1}/{len(self.entries)}")
        self.show_current()

    def _annotated_pixmap(self, pix: QPixmap, detections: List[Detection], current_num: Optional[str]) -> QPixmap:
        annotated = QPixmap(pix.size()); annotated.fill(Qt.black)
        painter = QPainter(annotated)
        painter.drawPixmap(0, 0, pix)

        pen_all = QPen(QColor(0, 200, 0), 3)
        pen_sel = QPen(QColor(255, 220, 0), 4)
        painter.setFont(QFont("Arial", 18))

        for det in detections:
            pts = [(int(det.box[i][0]), int(det.box[i][1])) for i in range(4)]
            is_sel = (det.text == (current_num or ""))
            painter.setPen(pen_sel if is_sel else pen_all)
            for i in range(4):
                x1, y1 = pts[i]; x2, y2 = pts[(i+1) % 4]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            painter.setPen(QPen(QColor(0,255,0), 1))
            tx = min(p[0] for p in pts); ty = min(p[1] for p in pts) - 6
            painter.drawText(int(tx), int(max(10, ty)), f"{det.text} {det.conf:.2f}")

        painter.end()
        return annotated

    def _maybe_remove_empty_number_dir(self, num: str):
        if not self.out_dir:
            return
        dstdir = self.out_dir / num
        try:
            if dstdir.exists() and dstdir.is_dir():
                try:
                    next(dstdir.iterdir())
                except StopIteration:
                    dstdir.rmdir()
        except Exception:
            pass

    def show_current(self):
        if not self.entries:
            return
        e = self.entries[self.img_idx]

        pix = QPixmap(str(e.path))
        if pix.isNull():
            pix = QPixmap(1024, 768); pix.fill(QColor(20,20,20))

        has_candidate = self.cand_idx >= 0 and self.cand_idx < len(e.candidates)
        current_num = e.candidates[self.cand_idx] if has_candidate else None
        annotated = self._annotated_pixmap(pix, e.detections, current_num)
        self.view.set_pixmap(annotated)

        total = len(self.entries)
        self.images_progress_label.setText(f"Imágenes: {self.img_idx+1}/{total}")
        self.info_label.setText(f"Archivo: {e.path.name}  ({self.img_idx+1}/{total})   Número actual: {current_num or '-'}")
        self.current_num_edit.setText(current_num or "")
        self.current_num_edit.setEnabled(has_candidate)
        if has_candidate:
            self.current_num_edit.setFocus()
            self.btn_confirm_missing.setStyleSheet("")
        else:
            self.missing_edit.setEnabled(True)
            self.missing_edit.setFocus()
            self.btn_confirm_missing.setStyleSheet("background:#22a022; color:white; font-weight:600;")

        self.btn_accept.setEnabled(has_candidate)
        self.btn_reject.setEnabled(has_candidate)
        self.missing_edit.setEnabled(not has_candidate)
        self.btn_confirm_missing.setEnabled(not has_candidate)

        self.accepted_list.clear()
        for n in sorted(self.accepted_paths.keys()):
            self.accepted_list.addItem(QListWidgetItem(n))

    def _copy_to_number(self, e: ImageEntry, num: str) -> Path:
        dstdir = (self.out_dir / num); dstdir.mkdir(exist_ok=True)
        dst = dstdir / e.path.name
        shutil.copy2(e.path, dst)
        return dst

    def _log_csv(self, e: ImageEntry, num: str, action: str, dst: str = ""):
        append_csv_row(self.csv_path,
            [readable_time(), e.path.name, e.path.as_posix(), num, action, dst,
             "|".join(d.text for d in e.detections)],
            header=["timestamp","filename","filepath","number","action","dst_path","detected_texts"])

    def accept_current(self):
        if not self.entries: return
        e = self.entries[self.img_idx]
        if self.cand_idx == -1 or self.cand_idx >= len(e.candidates):
            self.confirm_missing()
            return
        num = clean_number(self.current_num_edit.text())
        if not num:
            return
        if num in self.accepted_paths:
            self._advance_candidate(); return
        dst = self._copy_to_number(e, num)
        self.accepted_paths[num] = dst
        self._log_csv(e, num, "accepted", str(dst))
        self._advance_candidate()

    def reject_current(self):
        if not self.entries: return
        e = self.entries[self.img_idx]
        if self.cand_idx == -1 or self.cand_idx >= len(e.candidates):
            return
        num = e.candidates[self.cand_idx]
        self._log_csv(e, num, "rejected", "")
        self._advance_candidate()

    def _advance_candidate(self):
        self.show_current()
        e = self.entries[self.img_idx]
        self.cand_idx += 1
        if self.cand_idx >= len(e.candidates):
            self.cand_idx = -1
        self.show_current()

    def _on_item_clicked_delete(self, item: QListWidgetItem):
        if not item: return
        n = item.text()
        self._remove_number_from_list(n)

    def remove_selected_accepted(self):
        items = self.accepted_list.selectedItems()
        if not items: return
        self._remove_number_from_list(items[0].text())

    def _remove_number_from_list(self, n: str):
        p = self.accepted_paths.get(n)
        if p:
            try:
                Path(p).unlink(missing_ok=True)
            except TypeError:
                try:
                    if Path(p).exists():
                        Path(p).unlink()
                except Exception:
                    pass
            except Exception:
                pass
        e = self.entries[self.img_idx]
        self._log_csv(e, n, "undo_delete", "")
        self.accepted_paths.pop(n, None)
        self._maybe_remove_empty_number_dir(n)
        self.show_current()

    def confirm_missing(self):
        if not self.entries:
            return
        e = self.entries[self.img_idx]
        if self.cand_idx != -1 and self.cand_idx < len(e.candidates):
            return
        text = self.missing_edit.text().strip()
        nums = parse_numbers_free(text)
        if nums:
            added = False
            for n in nums:
                if n and n not in self.accepted_paths:
                    dst = self._copy_to_number(e, n)
                    self.accepted_paths[n] = dst
                    self._log_csv(e, n, "accepted_missing", str(dst))
                    added = True
            self.missing_edit.clear()
            self.show_current()
            if added:
                self.status_label.setText("✅ Faltante agregado — Enter vacío para pasar")
            return
        self.next_image()

    def skip_image(self):
        if not self.entries: return
        e = self.entries[self.img_idx]
        self._log_csv(e, "", "skipped_image", "")
        self.next_image()

    def next_image(self):
        self.accepted_paths.clear()
        if self.img_idx + 1 < len(self.entries):
            self.img_idx += 1
            self.cand_idx = 0 if self.entries[self.img_idx].candidates else -1
            self.images_progress_label.setText(f"Imágenes: {self.img_idx+1}/{len(self.entries)}")
            self.show_current()
        else:
            QMessageBox.information(self, "Fin", "Revisión finalizada.")
            self.status_label.setText("Estado: terminado.")
            self.images_progress_label.setText(f"Imágenes: {len(self.entries)}/{len(self.entries)}")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.btn_theme.setText("☀" if self.dark_mode else "🌙")
        self._apply_theme()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.cand_idx == -1:
                self.confirm_missing()
            else:
                self.accept_current()
        elif ev.key() == Qt.Key_Plus or ev.text() == '+':
            self.reject_current()
        elif ev.key() in (Qt.Key_N,):
            self.skip_image()
        elif ev.key() == Qt.Key_T and (ev.modifiers() & Qt.ControlModifier):
            self.toggle_theme()
        else:
            super().keyPressEvent(ev)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress:
            if ev.key() == Qt.Key_Plus or ev.text() == '+':
                self.reject_current()
                return True
        return super().eventFilter(obj, ev)

def main():
    app = QApplication(sys.argv)

    # --------- ICONO DEL PROGRAMA ----------
    icon_candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Real Vision OCR" / "assets" / "realvisionocr.ico",
        Path(__file__).resolve().parent / "assets" / "realvisionocr.ico",
    ]
    for p in icon_candidates:
        if p and p.exists():
            app.setWindowIcon(QIcon(str(p)))
            break
    # ---------------------------------------

    font = app.font(); font.setPointSize(10); app.setFont(font)
    w = App()
    # también lo aplico a la ventana por si acaso
    for p in icon_candidates:
        if p and p.exists():
            w.setWindowIcon(QIcon(str(p)))
            break
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
