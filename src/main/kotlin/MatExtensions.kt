import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

fun Mat.resizeIfNeeded(
    maxWidth: Int,
    maxHeight: Int
) {
    val scale = (maxWidth / this.width().toDouble())
        .coerceAtMost(maxHeight / this.height().toDouble())
        .coerceAtMost(1.0)
    Imgproc.resize(this, this, Size(0.0, 0.0), scale, scale)
}