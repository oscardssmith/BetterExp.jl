using Base.Math: significand_bits, exponent_bias

using Test, Printf

# the following compares the ulp between x and y.
# First it promotes them to the larger of the two types x,y
const infh(::Type{Float64}) = 1e300
const infh(::Type{Float32}) = 1e37
function countulp(T, x::AbstractFloat, y::AbstractFloat)
    X, Y = promote(x, y)
    x, y = T(X), T(Y) # Cast to smaller type
    (isnan(x) && isnan(y)) && return 0
    (isnan(x) || isnan(y)) && return 10000
    if isinf(x)
        (sign(x) == sign(y) && abs(y) > infh(T)) && return 0 # relaxed infinity handling
        return 10001
    end
    (x ==  Inf && y ==  Inf) && return 0
    (x == -Inf && y == -Inf) && return 0
    if y == 0
        x == 0 && return 0
        return 10002
    end
    if isfinite(x) && isfinite(y)
        return T(abs(X - Y) / ulp(y))
    end
    return 10003
end

const DENORMAL_MIN(::Type{Float64}) = 2.0^-1074 
const DENORMAL_MIN(::Type{Float32}) = 2f0^-149

function ulp(x::T) where {T<:AbstractFloat}
    x = abs(x)
    x == T(0.0) && return DENORMAL_MIN(T)
    val, e = frexp(x)
    return max(ldexp(T(1.0), e - significand_bits(T) - 1), DENORMAL_MIN(T))
end

countulp(x::T, y::T) where {T <: AbstractFloat} = countulp(T, x, y)
strip_module_name(f::Function) = last(split(string(f), '.')) # strip module name from function f

function test_acc(T, fun_table, xx, tol; debug = true, tol_debug = 5)
    @testset "accuracy $(strip_module_name(xfun))" for (xfun, fun) in fun_table
        rmax = 0.0
        rmean = 0.0
        xmax = map(zero, first(xx))
        tol_debug_failed = 0
        for x in xx
            q = xfun(x...)
            c = fun(map(BigFloat, x)...)
            u = countulp(T, q, c)
            rmax = max(rmax, u)
            xmax = rmax == u ? x : xmax
            rmean += u
            if debug && u > tol_debug
                tol_debug_failed += 1
                #@printf("%s = %.20g\n%s  = %.20g\nx = %.20g\nulp = %g\n", strip_module_name(xfun), q, strip_module_name(fun), T(c), x, u)
            end
        end
        if debug
            println("Tol debug failed $(100tol_debug_failed / length(xx))% of the time.")
        end
        rmean = rmean / length(xx)

        fmtxloc = isa(xmax, Tuple) ? string('(', join((@sprintf("%.5f", x) for x in xmax), ", "), ')') : @sprintf("%.5f", xmax)
        println(rpad(strip_module_name(xfun), 18, " "), ": max ", @sprintf("%f", rmax),
            rpad(" at x = " * fmtxloc, 40, " "),
            ": mean ", @sprintf("%f", rmean))
       
        t = @test trunc(rmax, digits=1) <= tol
    end
end

for (func, (basefunc, base)) in (myexp2=>(exp2,Val(2)), myexp=>(exp,Val(ℯ)), myexp10=>(exp10,Val(10)))
    tol = 1.5
    xx = range(MIN_EXP(base,Float64),  MAX_EXP(base,Float64), length = 10^6);
    fun_table = Dict(func => basefunc)
    test_acc(Float64, fun_table, xx, tol, debug = true, tol_debug = .5)

    xs = range(MIN_EXP(base,Float32),  MAX_EXP(base,Float32), length = 10^6);
    fun_table = Dict(func => basefunc)
    test_acc(Float32, fun_table, xs, tol, debug = true, tol_debug = .5)
end
