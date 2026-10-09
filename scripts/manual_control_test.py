"""Manual control test for JetRacer on JetPack 7 (steering servo = PCA9685 ch0, ESC = PCA9685 ch1).

Run on the Jetson with the wheels OFF the ground:
    ~/jr/bin/python manual_control_test.py center
    ~/jr/bin/python manual_control_test.py left
    ~/jr/bin/python manual_control_test.py right
    ~/jr/bin/python manual_control_test.py steering --seconds 15
    ~/jr/bin/python manual_control_test.py throttle

The PCA9685 keeps sending the last pulse after the script exits, so `left`/`right`/`center` hold their position.
"""
import argparse
import time

from jetracer.nvidia_racecar import NvidiaRacecar


def pulse_us(car, channel):
    pca = car.kit._pca
    return round(pca.channels[channel].duty_cycle / 65535 * 1e6 / pca.frequency)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=["center", "left", "right", "steering", "throttle"])
    parser.add_argument("--seconds", type=float, default=15.0, help="length of the steering sweep")
    parser.add_argument("--amount", type=float, default=0.7, help="steering sweep amplitude, 0-1")
    args = parser.parse_args()

    car = NvidiaRacecar()  # patched: sends center steering + neutral throttle at startup
    print(f"start: steering {pulse_us(car, 0)} us, throttle {pulse_us(car, 1)} us")

    if args.test == "center":
        return
    if args.test in ("left", "right"):
        car.steering = -1.0 if args.test == "left" else 1.0
        print(f"holding steering {car.steering:+.1f} -> {pulse_us(car, 0)} us")
        return

    try:
        if args.test == "steering":
            t0, n = time.time(), 0
            while time.time() - t0 < args.seconds:
                car.steering = -args.amount if n % 2 == 0 else args.amount
                n += 1
                time.sleep(1.0)
            print(f"{n} swings in {time.time() - t0:.1f}s")
        else:
            print("neutral 3 s (ESC arming)")
            time.sleep(3)
            for t in (0.2, 0.3):
                car.throttle = t
                print(f"throttle {t} -> {pulse_us(car, 1)} us (3 s)")
                time.sleep(3)
    finally:
        car.throttle = 0.0
        car.steering = 0.0
        print(f"stopped: steering {pulse_us(car, 0)} us, throttle {pulse_us(car, 1)} us")


if __name__ == "__main__":
    main()
