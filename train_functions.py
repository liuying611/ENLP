# We have to build a function that pad tha sequences that are less then the filter size
# The model will throw an error if a sentence/sequence is less then the kernel_size of certain conv, so we must pad it
# to the maximum size of kernel_size.
import torch



def cnn_padding_to_match_filtersize(text, filter_sizes):
    features_ = []
    for f in text:
        f = f.cpu().numpy().tolist()
        if len(f) < max(filter_sizes):
            f += [1] * (max(filter_sizes) - len(f))
            features_.append(f)
        else:
            features_.append(f)

    return torch.LongTensor(features_).to(device)