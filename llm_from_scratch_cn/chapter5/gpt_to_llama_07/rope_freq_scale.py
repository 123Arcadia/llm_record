import torch
import  matplotlib.pyplot as plt

d = 1024
i = torch.arange(0, d // 2 + 1)
base_llama2 = 10000
base_llama3 = 50000

# "-": 指数的a^(-b) = 1/ (a^(b))
theta_llama2 = base_llama2 ** (-2 * (i - 1) / d)
theta_llama3 = base_llama3 ** (-2 * (i - 1) / d)

plt.plot(i, theta_llama2, label="base = 10_000(llama2)")
plt.plot(i, theta_llama3, label="base = 50_000(llama3)", ls="--")
plt.xlabel("dimension index x (i ... d/2)")
plt.ylabel(r"$\theta_i = \mathrm{base}^{-2(i-1)/d}$")
plt.legend()
plt.tight_layout()
plt.show()