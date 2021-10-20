import torch
import pandas as pd


def get_children(model: torch.nn.Module):
    # get children form model
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child
        return model
    else:
        # look for children from children... to the last child
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def get_all_shape(model):
    children = get_children(model)
    all_input_shape = list()

    def make_forward(original_forward):
        def new_forward(x):
            all_input_shape.append(x.shape)
            out = original_forward(x)
            return out

        return new_forward

    for layer in children:
        original_forward = layer.forward
        layer.forward = make_forward(original_forward)

    data = torch.randn(1, 3, 224, 224)
    result = model(data)
    return all_input_shape


def prune_df(df):
    trash_list = list()

    for i in range(len(df)):
        if df.iloc[i]['Op'] in ['__add__', '__iadd__', '__mul__']:
            trash_list.append(i)
        else:
            pass

    for i in trash_list:
        df.at[i - 1,
              'duration'] = df.iloc[i]['duration'] + df.iloc[i - 1]['duration']
    df = df.drop(trash_list).reset_index()
    return df
