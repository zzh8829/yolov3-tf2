import valohai


def main(old_config):
    papi = valohai.Pipeline(name="train-superbai", config=old_config)

    # Define nodes
    convert = papi.execution("convert-superbai")
    weights = papi.execution("weights")
    train = papi.execution("train")
    detect = papi.execution("detect")

    # Configure pipeline
    convert.output("classes.txt").to(train.input("classes"))
    convert.output("train/*").to(train.input("train"))
    convert.output("test/*").to(train.input("test"))
    convert.output("classes.txt").to(detect.input("classes"))

    weights.output("model/*").to(train.input("model"))

    train.output("model/*").to(detect.input("model"))

    return papi
