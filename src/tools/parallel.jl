export Parallel, Serial, Thread, serial, thread


abstract type Parallel end

struct Serial <:Parallel end
struct Thread <:Parallel end

const serial = Serial()
const thread = Thread()
