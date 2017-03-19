import train as tr
import sample as sp
import data_loader
import model as md
import config


def main(infer):

    args = config.para_config()
    data = data_loader.Data('samp2.txt', args)
    model = md.Model(args, data, infer=infer)

    run_fn = sp.sample if infer else tr.train

    run_fn(data, model, args)


if __name__ == '__main__':
    msg = """
    Usage:
    Training:
        python3 gen_lyrics.py 0
    Sampling:
        python3 gen_lyrics.py 1
    """
    control = 1
    infer = int(control)
    print('--Sampling--' if infer else '--Training--')
    main(infer)