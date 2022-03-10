import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import java.awt.Dimension
import java.awt.Toolkit
import kotlin.math.abs
import kotlin.math.floor
import kotlin.system.exitProcess

const val WINDOW_NAME_ORIGINAL = "Hand detection"
const val WINDOW_NAME_PROCESSED = "Hand detection (processed)"
const val WINDOW_NAME_SPLIT = "Hand detection (split)"
const val WINDOW_NAME_ARROW = "Hand detection (arrow)"

data class PartData(
    val partNumber: Int,
    val partRatio: Double
)

data class SplitImageData(
    val splitImage: Mat,
    val partsData: List<PartData>
) {
    val largestPartIndex: Int
        get() {
            val maxPart = partsData.maxByOrNull { it.partRatio } ?: return -1
            val equalParts = partsData.filter { it.partRatio == maxPart.partRatio }.size
            return if (equalParts != 1) -1
            else maxPart.partNumber
        }

    val asAngle: Double
        get() {
            val largestPartNumber = partsData.size - 1
            val partAngle = 180.0 / largestPartNumber
            var sumAngle = 0.0

            for (partData in partsData) {
                sumAngle += abs(partData.partNumber - largestPartNumber) * partData.partRatio * partAngle
            }
            return sumAngle
        }
}

fun main() {
    OpenCV.loadLocally()

    val screenDimension = Toolkit.getDefaultToolkit().screenSize
    initWindows(screenDimension)

    val cap = VideoCapture(0, Videoio.CAP_DSHOW)
    val img = Mat()
    if (cap.isOpened) {
        while (true) {
            if (!cap.read(img)) continue
            processImage(img, screenDimension)

            val keyPressed = HighGui.waitKey(16)
            if (keyPressed == 27) break
        }
    }

    cap.release()

    HighGui.waitKey()
    HighGui.destroyAllWindows()

    exitProcess(0)
}

fun initWindows(
    screenDimension: Dimension
) {
    HighGui.namedWindow(WINDOW_NAME_ORIGINAL, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_ORIGINAL, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_ORIGINAL, 0, 0)

    HighGui.namedWindow(WINDOW_NAME_PROCESSED, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_PROCESSED, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_PROCESSED, 0, screenDimension.height / 2)

    HighGui.namedWindow(WINDOW_NAME_SPLIT, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_SPLIT, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_SPLIT, screenDimension.width / 2, screenDimension.height / 2)

    HighGui.namedWindow(WINDOW_NAME_ARROW, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_ARROW, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_ARROW, screenDimension.width / 2, 0)
}

fun processImage(
    src: Mat,
    screenDimension: Dimension
) {
    resizeImage(src, screenDimension.width / 2, screenDimension.height / 2)
    Core.flip(src, src, 1)
    HighGui.imshow(WINDOW_NAME_ORIGINAL, src)

    val handImg = keepHand(src)
    HighGui.imshow(WINDOW_NAME_PROCESSED, handImg)

    val splitImgData = splitImage(handImg, parts = 7)
    highlightPart(splitImgData.splitImage, splitImgData.largestPartIndex, splitImgData.partsData.size)
    HighGui.imshow(WINDOW_NAME_SPLIT, splitImgData.splitImage)

    val arrowImg = createArrowImage(splitImgData.asAngle)
    resizeImage(arrowImg, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.imshow(WINDOW_NAME_ARROW, arrowImg)
}

fun resizeImage(
    src: Mat,
    maxWidth: Int,
    maxHeight: Int
) {
    val scale = (maxWidth / src.width().toDouble())
        .coerceAtMost(maxHeight / src.height().toDouble())
        .coerceAtMost(1.0)
    Imgproc.resize(src, src, Size(0.0, 0.0), scale, scale)
}

fun keepHand(
    src: Mat,
    lowerColor: DoubleArray = doubleArrayOf(90.0, 105.0, 0.0),
    upperColor: DoubleArray = doubleArrayOf(110.0, 230.0, 255.0)
): Mat {
    val dst = Mat()

    Imgproc.GaussianBlur(src, dst, Size(3.0, 3.0), 0.0)
    Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2HSV)
    Core.inRange(dst, Scalar(lowerColor), Scalar(upperColor), dst)

    Imgproc.medianBlur(dst, dst, 5)
    Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(8.0, 8.0)).apply {
        Imgproc.dilate(dst, dst, this)
    }

    return dst
}

fun splitImage(
    src: Mat,
    parts: Int = 3,
    partColors: List<Scalar> = listOf(
        Scalar(0.0, 0.0, 255.0),
        Scalar(0.0, 255.0, 0.0),
        Scalar(255.0, 0.0, 0.0),
        Scalar(0.0, 255.0, 255.0),
        Scalar(255.0, 0.0, 255.0),
        Scalar(255.0, 255.0, 0.0),
        Scalar(255.0, 255.0, 255.0)
    )
): SplitImageData {

    if (partColors.size < parts) return SplitImageData(Mat(), emptyList())

    val dst = Mat()
    val masks = generatePartMasks(src, parts)

    val partsData = mutableListOf<PartData>()
    val totalPixels = Core.countNonZero(src)

    Imgproc.cvtColor(src, dst, Imgproc.COLOR_GRAY2BGR)

    for (j in 0 until parts) {
        val partImg = Mat().apply { src.copyTo(this, masks[j]) }
        dst.setTo(partColors[j], partImg)

        val partPixels = Core.countNonZero(partImg)
        partsData.add(PartData(j, partPixels / totalPixels.toDouble()))
    }

    return SplitImageData(dst, partsData)
}

fun highlightPart(
    src: Mat,
    part: Int,
    totalParts: Int,
    color: Scalar = Scalar(0.0, 69.0, 255.0)
) {
    if (part < 0 || part > totalParts) return

    val rects = generatePartRects(src, totalParts)
    Imgproc.rectangle(src, rects[part], color, 2)
}

fun generatePartMasks(
    src: Mat,
    parts: Int
): List<Mat> {
    val masks = mutableListOf<Mat>()
    val rects = generatePartRects(src, parts)

    for (i in 0 until parts) {
        val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
        Imgproc.rectangle(mask, rects[i], Scalar(255.0, 255.0, 255.0), -1)
        masks.add(mask)
    }

    return masks
}

fun generatePartRects(
    src: Mat,
    parts: Int
): List<Rect> {
    val rects = mutableListOf<Rect>()
    val intervals = splitInterval(0, src.cols() - 1, parts)

    for (i in 0 until parts) {
        val point1 = Point(intervals[i].first.toDouble(), 0.0)
        val point2 =
            Point(intervals[i].second.toDouble() + 1, src.height().toDouble()) // délkelet irányába 1 pixelnyi eltolás
        rects.add(Rect(point1, point2))
    }

    return rects
}

fun splitInterval(
    start: Int,
    end: Int,
    parts: Int
): List<Pair<Int, Int>> {
    val size = floor((end - start + 1) / parts.toDouble())
    var mod = (end - start + 1) % parts.toDouble()

    var nStart: Int
    var nEnd = start - 1

    val intervals = mutableListOf<Pair<Int, Int>>()

    for (i in 0 until parts) {
        nStart = nEnd + 1
        nEnd = (nStart + size - 1 + if (mod-- > 0) 1 else 0).toInt()
        intervals.add(Pair(nStart, nEnd))
    }

    return intervals
}

fun createArrowImage(
    angle: Double,
    paintColor: Scalar = Scalar(64.0, 64.0, 64.0),
    backgroundColor: Scalar = Scalar(255.0, 255.0, 255.0)
): Mat {
    val img = Mat(Size(600.0, 600.0), CvType.CV_8UC3, backgroundColor)
    val start = Point(300.0, 300.0)
    val end = Point(500.0, 300.0)

    if (angle.isNaN()) Imgproc.circle(img, start, 200, paintColor, 10)
    else Imgproc.arrowedLine(img, start, end.rotate(start, Math.toRadians(-angle)), paintColor, 10)

    return img
}
