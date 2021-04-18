from dist import Distributions

d = Distributions()

d.normal(0,10)

print(d.infer(100))