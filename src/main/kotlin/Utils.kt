import org.opencv.core.Scalar
import kotlin.math.floor

fun splitDiscreteInterval(
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

fun generateColorsInHsv(
    count: Int
): List<Scalar> {
    val colors = mutableListOf<Scalar>()
    val intervals = splitDiscreteInterval(0, 180, count)
    for (interval in intervals) {
        colors.add(Scalar(interval.first.toDouble(), 255.0, 255.0))
    }
    return colors
}