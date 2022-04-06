import org.opencv.core.Size
import org.opencv.highgui.HighGui
import java.awt.Dimension
import java.awt.Toolkit

const val WINDOW_NAME_ORIGINAL = "Hand detection"
const val WINDOW_NAME_PROCESSED = "Hand detection (processed)"
const val WINDOW_NAME_SPLIT = "Hand detection (split)"
const val WINDOW_NAME_ARROW = "Hand detection (arrow)"

val screenDimension: Dimension by lazy {
    Toolkit.getDefaultToolkit().screenSize
}

val sizeOfImage: Size by lazy {
    Size(screenDimension.width / 2.0, screenDimension.height / 2.0)
}

fun initApplicationWindows() {
    HighGui.namedWindow(WINDOW_NAME_ORIGINAL, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_ORIGINAL, sizeOfImage.width.toInt(), sizeOfImage.height.toInt())
    HighGui.moveWindow(WINDOW_NAME_ORIGINAL, 0, 0)

    HighGui.namedWindow(WINDOW_NAME_PROCESSED, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_PROCESSED, sizeOfImage.width.toInt(), sizeOfImage.height.toInt())
    HighGui.moveWindow(WINDOW_NAME_PROCESSED, 0, screenDimension.height / 2)

    HighGui.namedWindow(WINDOW_NAME_SPLIT, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_SPLIT, sizeOfImage.width.toInt(), sizeOfImage.height.toInt())
    HighGui.moveWindow(WINDOW_NAME_SPLIT, screenDimension.width / 2, screenDimension.height / 2)

    HighGui.namedWindow(WINDOW_NAME_ARROW, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_ARROW, sizeOfImage.width.toInt(), sizeOfImage.height.toInt())
    HighGui.moveWindow(WINDOW_NAME_ARROW, screenDimension.width / 2, 0)
}