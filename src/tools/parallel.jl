export Parallel, Serial, Thread, serial, thread

"""
    Parallel

An abstract type for the different kinds of parallel programming supported.
"""
abstract type Parallel end

"""
    Serial <: Parallel

A type for serial computations
"""
struct Serial <:Parallel end

"""
    Thread <: Parallel

A type for multi-threading computations
"""
struct Thread <:Parallel end

const serial = Serial()
const thread = Thread()
