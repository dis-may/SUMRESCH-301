
What does the NN take as input and what does it produce as output?
===============================

INPUT: 

1. Sensorimotor state
2. Action to-be-taken

3. Previous sensorimotor state or some other context that makes it possible to
   interpret the current state and action when the sensory input is ambiguous
   (optional)   

OUTPUT:

1. Sensorimotor state after that action is taken

Example data

it=n+0 : x, y, α, ls, rs, lm, rm
it=n+1 : x, y, α, ls, rs, lm, rm
it=n+2 : x, y, α, ls, rs, lm, rm

correct_output: ls, rs, lm, rm at it=n+2

input depends on when it is being determined, is lm at it_n what the motor was
to get there, or what the motor will be to get to the next state? I think the
latter. So assuming that...

input : ls, rs, lm, rm at it=n+1

so I first thought I needed to involve three time steps to predict the next, but
I only need two, because the motor commands at it=n+1 are what cause the
transition from it=n+1 to it=n+2.

JAN 9.

May want to have it predict *the change in sensorimotor state* rather than the actual
state, to make it focus on the changes which are almost always small.
