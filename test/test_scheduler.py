from generalframework.scheduler import RampScheduler

def test_rampfunction():
    l= []
    max_epoch, max_value, ramp_mult, last_epoch = 10,1,-5,-1
    scheduler = RampScheduler(max_epoch,max_value,ramp_mult,last_epoch)
    for i in range(20):
        scheduler.step()
        l.append(scheduler.value)

    import matplotlib.pyplot as plt

    plt.plot(l)
    plt.show()
if __name__ == '__main__':

    test_rampfunction()





