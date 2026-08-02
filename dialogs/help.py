# dialogs/help.py
"""Help and information dialogs."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextOption, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTextBrowser,
    QPushButton,
)
from ui.styles import PRIMARY_BLUE
from locales import trans
from utils import resource_path


class HelpDialog(QDialog):
    """Help-info dialog (extracted from main.py)"""

    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 400)

        # Apply the dark theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #1e1e1e;
            }}
            QTextBrowser {{
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
                font-family: 'Segoe UI', 'Microsoft YaHei';
                font-size: 13px;
                selection-background-color: {PRIMARY_BLUE};
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:pressed {{
                background-color: #333;
            }}
        """)

        layout = QVBoxLayout(self)

        # Create a scrollable text browser
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(content)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(self.text_browser)

        # Close button
        close_btn = QPushButton(trans.t('btn_close'))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # Center on screen
        if parent:
            self.move(
                parent.x() + parent.width() // 2 - self.width() // 2,
                parent.y() + parent.height() // 2 - self.height() // 2,
            )


class RenderMethodHelpDialog(HelpDialog):
    """Render-method help dialog"""

    def __init__(self, parent=None):
        help_text = """<h3>Render Methods</h3>
        
    <p>Guided Filter<br/>
    Guided-filter fusion tuned for practical edge preservation. Ideal for simpler scenes or moderate focus variations. The kernel slider sets the base/detail split, but this method reconstructs base + detail exactly, so the setting has almost no visible effect - leave it at the default unless you have a reason not to.</p>

    <p>DCT<br/>
    Frequency-domain fusion that measures the fine detail in each block of the frame and takes each region from the frames that resolve it. Broad shading is discounted, so a blurred bright area cannot pass for a sharp one. Three controls tune it: DCT block sets how fine the decisions are, Focus tolerance trades a crisper never-in-focus background against a smoother one, and Blend block seams can be turned off to copy pixels from single frames untouched. It is fast, fully CPU-based, and works well when you need crisp edges without deploying neural models.</p>

    <p>DTCWT<br/>
    Dual-tree complex wavelet fusion that decomposes the stack across scales and orientations before recombining it. It is well suited to intricate, high-frequency content where retaining fine detail is critical.</p>

    <p>GFG-FGF<br/>
    GFG-FGF is a multi-focus image fusion algorithm based on a generalized four-neighborhood Gaussian gradient (GFG) operator combined with a fast guided filter (FGF). Feature extraction uses the GFG operator to capture high-frequency edge and gradient information. Information enhancement leverages the FGF together with the original image texture to smooth defocused regions while emphasizing focused areas. The fusion strategy constructs a pixel-wise decision map by selecting the maximum focus measure per pixel and then refines these decisions with FGF for edge-preserving smoothing, producing a weighted fusion that favors sharp, well-focused pixels.</p>

    <p>Pyramid<br/>
    Laplacian-pyramid fusion: each frame is split into band-pass detail levels plus a low-frequency base. Every band is a weighted mean across the stack, each frame weighted by how far its pooled local energy falls behind the best on offer - a frame twice behind the winner contributes well under a percent, so a real focus decision is still a decision, while frames that tie (a background no frame resolves) are averaged rather than picked between. Energies are compared in units of each frame's own grain, so a bright noisy frame cannot win the areas that hold no detail, and the low-frequency base follows the frames that carried the detail. The collapsed result is finally held inside the range its own frames span, so bands taken from different frames cannot add up to a value no frame had. The kernel slider sets the energy pooling window; the Pyramid group below it exposes the rest, all defaulted to their measured best, including the textbook choose-max rule under Selectivity. Runs on CPU or GPU.</p>

    <p>Depth Map (Max)<br/>
    Classic hard depth-map fusion: a local Laplacian-energy focus measure is computed per frame, and each pixel is taken whole from the frame where that measure is highest. The decision is an order-independent per-pixel argmax, so colour and noise stay clean within a slice and never blend across sources. The kernel slider sets the window the focus measure is pooled over - larger is steadier on noise, smaller follows finer detail. Fully CPU-based.</p>

    <p>Depth Map (Average)<br/>
    Contrast-weighted average: frames are blended in proportion to the same local focus measure rather than hard-selected. Sharp detail is still dominated by the frame that holds it, but a flat region - where every frame is equally focused - collapses to the plain mean, recovering the stack's multi-frame signal-to-noise (a free sqrt(N) noise reduction). Use it on stacks with large smooth areas where Max would chase sensor noise. The kernel slider sets the focus-measure window. Fully CPU-based.</p>

    <p>StackMFF-V4<br/>
    A neural network trained on everyday focus stacks. It generally produces the strongest results with minimal tuning. Because it is not fine-tuned for specialist domains (microphotography, microscopy, medical imaging, etc.), avoid it when domain shifts are expected. Runs fastest with GPU acceleration.</p>

    <p>+ IFCNN Refine<br/>
    An optional stage that runs after the selected fusion method rather than replacing it. The fused candidate and the aligned source stack are encoded by IFCNN and merged in feature space, so detail the fusion step missed - blur bleeding across edges, pixels taken from the wrong slice - is recovered from whichever source frame actually holds it. Requires PyTorch and the IFCNN weights file in the weights folder; large images are processed tile by tile.</p>"""

        super().__init__(trans.t('help_render_title'), help_text, parent)


class RegistrationHelpDialog(HelpDialog):
    """Registration-method help dialog"""

    def __init__(self, parent=None):
        help_text = """<h3>Registration Methods</h3>

    <p>Scale (focus breathing)<br/>
    Corrects the magnification change a lens introduces as the focus plane moves through a stack&mdash;"focus breathing". Fits a constrained similarity (uniform scale plus a small rotation and recentring) from SIFT matches, so it cancels the size drift without the overfitting a full homography risks on blurred frames. Enable it when frames grow or shrink slightly from first to last.</p>

    <p>Align (Homography)<br/>
    Feature-based alignment for stacks that need global geometric correction. Detects SIFT features between consecutive frames and fits both a perspective transform and a constrained similarity to each pair, keeping the perspective one only where its extra freedom predicts matches it was not fitted to. A stack shot on a rail carries no perspective, so its pairs take the constrained fit; a handheld shot where the camera really tilted keeps the full one.</p>

    <p>Align (ECC)<br/>
    Enhanced Correlation Coefficient alignment refines alignment at the sub-pixel level. Works well for fine adjustments or whenever feature detection is unreliable.</p>

    <p>All options are independent. When several are enabled they run in order&mdash;Scale first as a coarse magnification correction, then Homography, then ECC to refine the residual translation and rotation.</p>"""
        
        super().__init__(trans.t("help_registration_title"), help_text, parent)


class TileHelpDialog(HelpDialog):
    """Tile-parameter help dialog"""

    def __init__(self, parent=None):
        help_text = """<h3>Tile Settings Help</h3>
        <p>tile_enabled: Enable or disable tiled processing. When enabled, large images
        will be processed in smaller blocks to reduce memory usage.</p>

        <p>tile_block_size: Size (in pixels) of each square tile block. Typical values
        are 512–2048 depending on memory and speed tradeoffs.</p>

        <p>tile_overlap: Overlap (in pixels) between adjacent tiles used to avoid seams
        when combining results. A positive overlap helps smooth boundaries.</p>

        <p>tile_threshold: If the image's longest side is larger than this threshold,
        tiled processing will be considered. Smaller images are processed as a whole.</p>"""

        super().__init__(trans.t("help_tile_title"), help_text, parent)
