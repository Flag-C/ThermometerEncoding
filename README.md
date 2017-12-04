# ThermometerEncoding
This is a repo trying to reproduce [Thermometer Encoding: One Hot Way To Resist Adversarial Examples](https://openreview.net/forum?id=S18Su--CW) in pytorch. 

### Results on CIFAR10

I use ResNet-50 to reproduce the experiment instead of a Wide-ResNet. All LS-PGA attack is 7-step iterative white-box attack. However, I find that if I increase the attack step-size, attack failure rate will drop dramatically.

|               | clean  | LS-PGA $$\xi=0.01$$ | LS-PGA $$\xi=0.1$$ | LS-PGA $$\xi=1$$ | results on the paper(LS-PGA) | results on the paper(clean) |
| ------------- | ------ | ------------------- | ------------------ | ---------------- | ---------------------------- | --------------------------- |
| clean trained | 91.52% | 43.27%              | 3.14%              | 0.12%            | 50.50%                       | 94.22%                      |
| adv trained   | 89.75% | 74.00%              | 27.44%             | 15.02%           | 79.16%                       | 89.88%                      |