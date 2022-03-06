import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import java.awt.Dimension
import java.awt.Toolkit
import kotlin.math.floor

const val WINDOW_NAME_ORIGINAL = "Hand detection"
const val WINDOW_NAME_PROCESSED = "Hand detection (processed)"
const val WINDOW_NAME_SPLIT = "Hand detection (split)"
const val WINDOW_NAME_ARROW = "Hand detection (arrow)"

enum class Directions {
    STAY,

    NORTH,
    EAST,
    SOUTH,
    WEST
}

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

    val toDirection: Directions
        get() = when {
            partsData.size != 3 -> Directions.STAY
            largestPartIndex == 0 -> Directions.WEST
            largestPartIndex == 1 -> Directions.NORTH
            largestPartIndex == 2 -> Directions.EAST
            else -> Directions.STAY
        }
}

fun main(args: Array<String>) {
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

//    for (i in 1..28) {
//        val img = Imgcodecs.imread("./HandPictures/$i.jpg", Imgcodecs.IMREAD_COLOR)
//        processImage(img, screenDimension)
//
//        HighGui.waitKey()
//    }

    HighGui.waitKey()
    HighGui.destroyAllWindows()
}

fun initWindows(screenDimension: Dimension) {
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

fun processImage(src: Mat, screenDimension: Dimension) {
    resizeImage(src, screenDimension.width / 2, screenDimension.height / 2)
    Core.flip(src, src, 1)
    HighGui.imshow(WINDOW_NAME_ORIGINAL, src)

    val processedImg = keepHand(src)
    HighGui.imshow(WINDOW_NAME_PROCESSED, processedImg)

    val splitImgData = splitImage(processedImg)
    highlightPart(splitImgData.splitImage, splitImgData.largestPartIndex, splitImgData.partsData.size)
    HighGui.imshow(WINDOW_NAME_SPLIT, splitImgData.splitImage)

    val arrowImg = createDirectionImage(splitImgData.toDirection)
    resizeImage(arrowImg, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.imshow(WINDOW_NAME_ARROW, arrowImg)
}

fun resizeImage(src: Mat, maxWidth: Int, maxHeight: Int) {
    val scale = (maxWidth / src.width().toDouble())
        .coerceAtMost(maxHeight / src.height().toDouble())
        .coerceAtMost(1.0)
    Imgproc.resize(src, src, Size(0.0, 0.0), scale, scale)
}

fun keepHand(src: Mat): Mat {
    val lowerColor = doubleArrayOf(0.0, 58.0, 50.0)
    val upperColor = doubleArrayOf(30.0, 255.0, 255.0)

    val blur = Mat()
    val hsv = Mat()
    val mask = Mat()
    val dst = Mat()

    Imgproc.GaussianBlur(src, blur, Size(3.0, 3.0), 0.0)
    Imgproc.cvtColor(blur, hsv, Imgproc.COLOR_BGR2HSV)
    Core.inRange(hsv, Scalar(lowerColor), Scalar(upperColor), mask)

    Imgproc.medianBlur(mask, dst, 5)
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(8.0, 8.0))
    Imgproc.dilate(dst, dst, kernel)

    return dst
}

fun splitImage(
    src: Mat, parts: Int = 3, partColors: List<Scalar> = listOf(
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

    val splitImg = src.clone()
    val masks = generatePartMasks(src, parts)

    val partsData = mutableListOf<PartData>()
    val totalPixels = Core.countNonZero(src)

    Imgproc.cvtColor(splitImg, splitImg, Imgproc.COLOR_GRAY2BGR)

    for (j in 0 until parts) {
        val partImg = Mat().apply { src.copyTo(this, masks[j]) }
        splitImg.setTo(partColors[j], partImg)

        val partPixels = Core.countNonZero(partImg)
        partsData.add(PartData(j, partPixels / totalPixels.toDouble()))
    }

    return SplitImageData(splitImg, partsData)
}

fun highlightPart(src: Mat, part: Int, totalParts: Int, color: Scalar = Scalar(0.0, 69.0, 255.0)) {
    if (part < 0 || part > totalParts) return

    val rects = generatePartRects(src, totalParts)
    Imgproc.rectangle(src, rects[part], color, 2)
}

fun generatePartMasks(src: Mat, parts: Int): List<Mat> {
    val masks = mutableListOf<Mat>()
    val rects = generatePartRects(src, parts)

    for (i in 0 until parts) {
        val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
        Imgproc.rectangle(mask, rects[i], Scalar(255.0, 255.0, 255.0), -1)
        masks.add(mask)
    }

    return masks
}

fun generatePartRects(src: Mat, parts: Int): List<Rect> {
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

fun splitInterval(start: Int, end: Int, parts: Int): List<Pair<Int, Int>> {
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

fun createDirectionImage(
    direction: Directions,
    paintColor: Scalar = Scalar(64.0, 64.0, 64.0),
    backgroundColor: Scalar = Scalar(255.0, 255.0, 255.0)
): Mat {
    val img = Mat(Size(600.0, 400.0), CvType.CV_8UC3, backgroundColor)

    val start = when (direction) {
        Directions.EAST -> Point(100.0, 200.0)
        Directions.NORTH -> Point(300.0, 300.0)
        Directions.SOUTH -> Point(300.0, 100.0)
        Directions.WEST -> Point(500.0, 200.0)
        Directions.STAY -> Point(300.0, 200.0)
    }

    val end = when (direction) {
        Directions.EAST -> Point(500.0, 200.0)
        Directions.NORTH -> Point(300.0, 100.0)
        Directions.SOUTH -> Point(300.0, 300.0)
        Directions.WEST -> Point(100.0, 200.0)
        Directions.STAY -> Point(300.0, 200.0)
    }

    if (direction == Directions.STAY) Imgproc.circle(img, start, 100, paintColor, 10)
    else Imgproc.arrowedLine(img, start, end, paintColor, 10)

    return img
}

fun Rect.toMatOfPoints(): MatOfPoint {
    val tr = Point((x + width).toDouble(), y.toDouble())
    val bl = Point(x.toDouble(), (y + height).toDouble())

    return MatOfPoint(tl(), tr, br(), bl)
}