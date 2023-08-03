import time
import matplotlib.pyplot as plt

class PID:
    def __init__(self, setpoint=0.0, P=0.0, I=0.0, D=0.0, currentTime=None, satMin=-1.0, satMax=1.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        if currentTime == None:
            currentTime = time.time()
        self.currTime = currentTime
        self.prevTime = currentTime

        self.currError = 0.0
        self.prevError = 0.0

        self.saturationMin = satMin
        self.saturationMax = satMax

        self.setpoint = setpoint

        self.u = 0.0
        self.prevU = 0.0
        self.antiwindup = 0.0

        self.Yp = 0.0
        self.Yi = 0.0
        self.Yd = 0.0

    def output(self, feedback=0.0, currentTime=None):

        # count regulation error
        self.currError = self.setpoint - feedback  # negative feedback

        # set current time
        if currentTime is None:
            self.currTime = time.time()
        else:
            self.currTime = currentTime

        # regulator uses trapezoid aproximation
        self.Yp = self.Kp * self.currError
        self.Yi += self.Ki * (self.currTime - self.prevTime) * (self.currError + self.prevError - self.antiwindup)/2  #((self.currError) + self.prevError)/2 - self.antiwindup
        self.Yd = self.Kd * (self.currError - self.prevError) / (self.currTime - self.prevTime)
        self.u = self.Yp + self.Yi + self.Yd

        self.prevU = self.u

        # saturation
        if self.u > self.saturationMax:
            self.u = self.saturationMax
        elif self.u < self.saturationMin:
            self.u = self.saturationMin

        # anti windup
        if self.Ki != 0.0:
            Tr = self.Ki/self.Kp
            self.antiwindup = (self.prevU - self.u)/Tr

        self.prevTime = self.currTime
        self.prevError = self.currError


        return self.u

    def update_params(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D

    def reset(self):
        self.u = 0.0
        self.prevU = 0.0
        self.antiwindup = 0.0

        self.Yp = 0.0
        self.Yi = 0.0
        self.Yd = 0.0
        self.currentTime = time.time()
        self.prevTime = self.currentTime


if __name__ == "__main__":
    dt = 0.00001
    steps = 100
    t = [i for i in range(steps)]
    w = [2.0 if (i > 20) and (i <= 50) else 0 for i in range(steps)]
    u = [0]  # init value
    e = []
    p = []
    i = []
    d = []
    S = 0
    R1 = PID(setpoint=0.0, P=0.66, I=10.0, currentTime=time.time())
    time.sleep(dt)
    for w_in in w:
        p.append(R1.Yp)
        i.append(R1.Yi)
        d.append(R1.Yd)
        e.append(w_in+S)
        reg = R1.output(feedback=e[-1])
        S+=reg
        u.append(reg)
        time.sleep(dt)
    # print(R1.Yi, R1.Yp, R1.antiwindup, R1.setpoint, R1.currError)
    plt.plot(t, w, label="w")
    plt.plot(t, u[:-1], label="u")
    plt.plot(t, e, label="e")
    plt.plot(t, p, label="p")
    plt.plot(t, i, label="i")
    plt.plot(t, d, label="d")
    plt.legend()
    plt.show()
