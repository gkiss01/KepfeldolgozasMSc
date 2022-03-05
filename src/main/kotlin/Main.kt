import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.awt.Toolkit
import kotlin.math.floor

const val WINDOW_NAME_ORIGINAL = "Hand detection"
const val WINDOW_NAME_PROCESSED = "Hand detection (processed)"
const val WINDOW_NAME_SPLIT = "Hand detection (split)"

fun main(args: Array<String>) {
    OpenCV.loadLocally()

    val dimension = Toolkit.getDefaultToolkit().screenSize

    HighGui.namedWindow(WINDOW_NAME_ORIGINAL, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_ORIGINAL, dimension.width / 2, dimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_ORIGINAL, 0, 0)

    HighGui.namedWindow(WINDOW_NAME_PROCESSED, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_PROCESSED, dimension.width / 2, dimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_PROCESSED, 0, dimension.height / 2)

    HighGui.namedWindow(WINDOW_NAME_SPLIT, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_SPLIT, dimension.width / 2, dimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_SPLIT, dimension.width / 2, dimension.height / 2)

    for (i in 1..28) {
        val img = Imgcodecs.imread("./HandPictures/$i.jpg", Imgcodecs.IMREAD_COLOR)

        resizeImage(img, dimension.width / 2, dimension.height / 2)
        HighGui.imshow(WINDOW_NAME_ORIGINAL, img)

        val processedImg = img.clone()
        keepHand(processedImg)
        HighGui.imshow(WINDOW_NAME_PROCESSED, processedImg)
        //HighGui.waitKey()

        val splitImg = processedImg.clone()
        val totalPixels = Core.countNonZero(splitImg)
        val masks = generatePartMasks(splitImg, 3)

        for (j in 0 until 3) {
            val partImg = Mat().apply { splitImg.copyTo(this, masks[j]) }
            val partPixels = Core.countNonZero(partImg)

            val partImgCol = Mat().apply { Imgproc.cvtColor(partImg, this, Imgproc.COLOR_GRAY2BGR) }
            partImgCol.setTo(Scalar(255.0, 0.0, 0.0), partImg)

            println("Part $j, pixels: $partPixels, total: $totalPixels")

            HighGui.imshow(WINDOW_NAME_SPLIT, partImgCol)
            HighGui.waitKey()
        }
    }

    HighGui.waitKey()
    HighGui.destroyAllWindows()
}

fun resizeImage(src: Mat, maxWidth: Int, maxHeight: Int) {
    val scale = (maxWidth / src.width().toDouble())
        .coerceAtMost(maxHeight / src.height().toDouble())
        .coerceAtMost(1.0)
    Imgproc.resize(src, src, Size(0.0, 0.0), scale, scale)
}

fun keepHand(src: Mat) {
    val lowerColor = doubleArrayOf(0.0, 58.0, 50.0)
    val upperColor = doubleArrayOf(30.0, 255.0, 255.0)

    val blur = Mat()
    val hsv = Mat()
    val mask = Mat()

    Imgproc.GaussianBlur(src, blur, Size(3.0, 3.0), 0.0)
    Imgproc.cvtColor(blur, hsv, Imgproc.COLOR_BGR2HSV)
    Core.inRange(hsv, Scalar(lowerColor), Scalar(upperColor), mask)

    Imgproc.medianBlur(mask, src, 5)
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(8.0, 8.0))
    Imgproc.dilate(src, src, kernel)
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

fun Rect.toMatOfPoints(): MatOfPoint {
    val tr = Point((x + width).toDouble(), y.toDouble())
    val bl = Point(x.toDouble(), (y + height).toDouble())

    return MatOfPoint(tl(), tr, br(), bl)
}