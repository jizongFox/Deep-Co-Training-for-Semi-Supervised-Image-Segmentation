from generalframework.scheduler import RampScheduler, ConstantScheduler, RampDownScheduler


def test_rampfunction():
    l = []
    begin_epoch, max_epoch, max_value, ramp_mult = 10, 100, 0.005, -5
    scheduler = RampScheduler(begin_epoch, max_epoch, max_value, ramp_mult)
    for i in range(120):
        scheduler.step()
        l.append(scheduler.value)

    import matplotlib.pyplot as plt

    plt.plot(l)
    plt.show()


def test_constantfunction():
    l = []
    begin_epoch, max_value, = 20, 1
    scheduler = ConstantScheduler(begin_epoch, max_value)
    for i in range(120):
        scheduler.step()
        l.append(scheduler.value)

    import matplotlib.pyplot as plt

    plt.plot(l)
    plt.show()


def test_rampDownScheduler():
    l = []
    max_epoch, max_value, ramp_mult, min_val, cutoff = 100, 0.01, -2, 0.005, 30
    scheduler = RampDownScheduler(max_epoch, max_value, ramp_mult, min_val, cutoff)
    for i in range(120):
        scheduler.step()
        l.append(scheduler.value)

    import matplotlib.pyplot as plt

    plt.plot(l)
    plt.show()


if __name__ == '__main__':
    test_rampfunction()
    # test_constantfunction()
    # test_rampDownScheduler()
