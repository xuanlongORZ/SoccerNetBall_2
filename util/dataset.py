from util.io import load_text


DATASETS = [
    'soccernetball'
]

def load_classes(file_name, event_team = False):
    if event_team:
        classes = {}
        for i, x in enumerate(load_text(file_name)):
            classes[x+'-left'] = (i*2)+1
            classes[x+'-right'] = (i*2)+2
        return classes
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}