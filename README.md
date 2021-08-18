# binomial-pricing

An implementation of the binomial option pricing model, including the CRR method, and various other methods.
The implementation is non-recombining, to allow for the pricing of options on non-recombining trees. The author acknowledges that a more efficient implementation, where the number of nodes at step i is i + 1, would allow for the pricing of options as dt -> 0 for binomial trees generated through methods akin to CRR.
In addition, the master file trees.py, includes different methods:
- Barrier Options
- Delta Trees
- When to exercise american options
- etc ...