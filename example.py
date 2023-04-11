import StrongPandas as s
import matplotlib.pyplot as plt

RfKf = s.constant(900 * 3e4)
Kb = s.constant(4.5e4)
Vb = s.Variable('输出电压幅值（硅光）/V')
Vf = s.Variable('输出电压幅值（热释电）/V')
l = s.Variable('波长/nm')
R = (Vb / Kb) / (Vf / RfKf)
R.asname('光谱响应度（硅光）')

s.get_var_from_csv('data.csv', [l, Vb, Vf], encoding='ansi')
R.eval()
# l.draw(Vf, Vb)
l.draw(R)
plt.show()
