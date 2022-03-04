import nu.pattern.OpenCV
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.awt.Toolkit

const val WINDOW_NAME = "Hand detection"

fun main(args: Array<String>) {
    OpenCV.loadLocally()

    val dimension = Toolkit.getDefaultToolkit().screenSize

    HighGui.namedWindow(WINDOW_NAME, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME, dimension.width / 2, dimension.height / 2)

    for (i in 1..28) {
        val img = Imgcodecs.imread("./HandPictures/$i.jpg", Imgcodecs.IMREAD_COLOR)

        resizeImage(img, dimension.width / 2, dimension.height / 2)
        keepHand(img)

        HighGui.imshow(WINDOW_NAME, img)
        HighGui.waitKey()
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