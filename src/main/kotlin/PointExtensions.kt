import org.opencv.core.Point
import kotlin.math.cos
import kotlin.math.sin

fun rotatePoint(point: Point, angRad: Double): Point {
    return Point().apply {
        x = point.x * cos(angRad) - point.y * sin(angRad)
        y = point.x * sin(angRad) - point.y * cos(angRad)
    }
}

fun rotatePoint(point: Point, center: Point, angRad: Double) = rotatePoint(point - center, angRad) + center

operator fun Point.plus(other: Point) = Point(x + other.x, y + other.y)
operator fun Point.minus(other: Point) = Point(x - other.x, y - other.y)

fun Point.rotate(angRad: Double) = rotatePoint(this, angRad)
fun Point.rotate(center: Point, angRad: Double) = rotatePoint(this, center, angRad)