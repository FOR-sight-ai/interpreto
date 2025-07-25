# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Common fixtures for all tests
"""

import torch
from pytest import fixture
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer

from interpreto import ModelWithSplitPoints
from interpreto.typing import LatentActivations


@fixture(scope="session")
def sentences():
    return [
        "Interpreto is the latin for 'to interpret'. But it also sounds like a spell from the Harry Potter books.",
        "Interpreto is magical",
        "Testing interpreto",
    ]


@fixture(scope="session")
def multi_split_model() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "hf-internal-testing/tiny-random-bert",
        split_points=[
            "cls.predictions.transform.LayerNorm",
            "bert.encoder.layer.1",
            "bert.encoder.layer.3.attention.self.query",
        ],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
        batch_size=4,
    )


@fixture(scope="session")
def splitted_encoder_ml() -> ModelWithSplitPoints:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ModelWithSplitPoints(
        "hf-internal-testing/tiny-random-bert",
        split_points=["bert.encoder.layer.1.output"],
        model_autoclass=AutoModelForSequenceClassification,  # type: ignore
        batch_size=4,
        device_map=device,
    )


@fixture(scope="session")
def activations_dict(splitted_encoder_ml: ModelWithSplitPoints, sentences: list[str]) -> dict[str, LatentActivations]:
    return splitted_encoder_ml.get_activations(
        sentences, activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN
    )  # type: ignore


@fixture(scope="session")
def bert_model():
    return AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert")


@fixture(scope="session")
def bert_tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")


@fixture(scope="session")
def real_bert_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@fixture(scope="session")
def gpt2_model():
    return AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")


@fixture(scope="session")
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")


@fixture(scope="session")
def huge_text():
    text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc tincidunt est eros. Aenean vitae metus dictum, pellentesque tortor et, maximus nibh. Phasellus suscipit massa nec enim auctor venenatis. Duis lacus sem, semper a augue sollicitudin, sollicitudin pulvinar arcu. Suspendisse velit neque, vulputate sit amet est varius, aliquam elementum nisl. Nunc at mi lobortis, volutpat diam non, volutpat mi. Nam ullamcorper, sapien eget efficitur aliquam, turpis neque lacinia leo, id venenatis risus nibh sed diam.

    Phasellus velit quam, blandit quis tempus id, varius vitae nisl. Vivamus neque sapien, laoreet non tortor sit amet, egestas dignissim sem. Quisque nec felis a augue tempus laoreet ut nec odio. Nulla facilisi. Nunc eget aliquam sem. Sed sit amet neque tincidunt, volutpat elit in, malesuada nisl. Morbi rhoncus, arcu efficitur euismod posuere, sem mauris pulvinar tellus, at ornare augue nulla eget nisi. Pellentesque et lectus vehicula, luctus leo ut, facilisis nulla. Fusce ullamcorper urna in leo tincidunt interdum. Vivamus at tincidunt justo, ut suscipit neque. Ut elementum dui risus, in vestibulum ex lobortis posuere.

    Fusce a faucibus purus. Nunc ac blandit elit, ut vulputate ligula. In blandit laoreet felis ut vulputate. Vivamus posuere fringilla velit, quis congue arcu porttitor vel. Integer facilisis lectus vitae nunc dapibus, at facilisis ante convallis. Fusce interdum lorem vel tellus luctus porta. Morbi placerat magna sed ipsum accumsan consectetur. Nullam nec congue erat. Mauris semper erat in suscipit congue. Aliquam iaculis sapien elit, et malesuada ex accumsan et. Nunc eget luctus mauris. Maecenas in commodo mauris. Vestibulum pellentesque nisl tortor, non pharetra lorem feugiat at. Quisque in sem varius, vehicula neque at, porttitor metus. Nunc augue felis, vestibulum non tortor quis, lobortis finibus arcu.

    Suspendisse eget felis egestas, maximus quam sed, aliquet ligula. Maecenas semper magna non turpis consectetur placerat sit amet a leo. Donec ipsum turpis, fringilla eget vulputate eget, sagittis ut urna. Ut bibendum purus velit, et placerat tortor gravida et. Sed odio turpis, facilisis ut commodo ac, pulvinar eget quam. Aenean sodales mauris scelerisque velit tristique imperdiet. In vitae lectus in metus tincidunt lacinia. In posuere arcu id nunc eleifend pharetra. Quisque nibh ipsum, vulputate pulvinar est in, iaculis vehicula nulla. Aliquam euismod urna ac risus viverra, ac efficitur turpis cursus. Etiam rhoncus purus sed mauris efficitur facilisis. Proin sit amet feugiat ligula. In efficitur dui et nulla accumsan luctus. Duis a lacus iaculis, dapibus arcu quis, mattis lorem.

    Ut est quam, convallis ut sollicitudin lacinia, hendrerit in metus. In consectetur elit quis lacus lacinia rutrum. Ut sodales felis ac euismod imperdiet. Aliquam auctor, elit quis eleifend fringilla, orci felis blandit arcu, vitae laoreet turpis nunc eget libero. Proin eget lectus efficitur, volutpat felis in, posuere ante. Aenean eget congue erat. Integer vel est nunc. Aliquam pellentesque, magna et convallis ullamcorper, diam dui semper lorem, nec posuere dui turpis quis turpis. Maecenas ultrices fringilla augue ut dictum. Mauris eu sagittis massa, vitae sollicitudin nulla. Sed gravida lacus vel massa ultricies, at rhoncus arcu pulvinar. Fusce nec luctus sapien, eu dignissim velit. Praesent tincidunt, nulla convallis ultrices aliquam, ante sapien vulputate quam, at varius ante libero commodo magna. Fusce quis interdum nisl, at blandit massa. Duis eu tincidunt sem, id bibendum sem.

    Curabitur ac placerat velit. Aenean faucibus enim sit amet vulputate lacinia. Quisque quis massa id dolor scelerisque euismod. Maecenas pellentesque tortor vel sapien fermentum, vel pretium tortor ullamcorper. Morbi ut velit nunc. Vestibulum porta sem nec enim finibus, quis lacinia ipsum convallis. Nunc viverra lacus in ligula molestie, ac maximus arcu elementum. Aenean aliquet interdum ultricies. Integer quis tincidunt est, ut feugiat quam. In rhoncus, orci ac semper porttitor, eros erat facilisis ante, a facilisis mauris ipsum sit amet lectus. Phasellus elementum mollis nisl, eget blandit arcu eleifend posuere. Aenean tincidunt sem vel feugiat tempus. Donec placerat sem eu ipsum tempus ornare.

    Mauris auctor, ipsum sit amet viverra accumsan, diam arcu lacinia orci, ut interdum nulla enim gravida ante. Integer odio arcu, maximus at ultrices eu, euismod ac tellus. Proin augue augue, iaculis id dui mattis, sollicitudin sollicitudin ante. Maecenas vitae diam nec dui bibendum auctor. Quisque sit amet scelerisque tortor, et dapibus nunc. Phasellus tortor orci, tempus et ex sed, tincidunt vestibulum ipsum. Nulla facilisi. Cras luctus dolor sit amet lacus commodo, quis fringilla ligula molestie. Donec placerat volutpat tincidunt. In hac habitasse platea dictumst. Cras bibendum vitae nibh sed iaculis. Ut dapibus massa id urna ornare egestas. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Fusce et convallis velit, ac accumsan diam. Suspendisse a efficitur tortor.

    Proin ac tristique velit, a tristique dui. Aliquam feugiat porta ex, id iaculis nulla vestibulum non. Morbi condimentum tristique nisi, ac suscipit dui fermentum eget. Maecenas rhoncus elit vel augue consectetur bibendum. Etiam faucibus imperdiet mi at vulputate. Mauris nec sem nisl. Nulla condimentum in libero et feugiat. Suspendisse ex arcu, mollis sit amet tortor non, porttitor tempus lectus. Vestibulum id ex bibendum, scelerisque est id, placerat neque. Etiam ut tristique mi, a iaculis libero.

    Suspendisse imperdiet tristique consequat. Proin et tellus semper, ullamcorper ex sed, dignissim nisl. Vestibulum id nunc id metus pellentesque auctor. Nullam lectus diam, efficitur non mauris sit amet, efficitur sagittis sem. Vivamus laoreet ligula non euismod pellentesque. Vivamus tempus mollis congue. Maecenas at dapibus ex. Aenean eget diam diam. Sed magna dolor, dignissim id dui eu, dignissim dapibus massa. Donec quis pulvinar nisl. Etiam malesuada, diam ac porta gravida, nibh mauris fermentum nisl, ac euismod orci nisi vel erat. Nullam nisi diam, dictum eu lacus a, cursus consequat erat.

    Nulla massa ante, tempus sit amet orci non, sollicitudin bibendum nibh. Mauris rutrum congue placerat. Donec auctor, enim et auctor volutpat, mi risus tempus ante, sed condimentum nunc risus ut purus. Fusce consequat sed neque ac blandit. Phasellus efficitur suscipit risus quis fermentum. Praesent nec posuere eros. Nunc iaculis libero id vulputate pretium.

    Sed porta urna vel mattis lacinia. Nam ultricies fermentum est eget molestie. Duis et fringilla nisl. Nullam semper mauris eu mollis convallis. Maecenas vehicula, nulla ut molestie pellentesque, mauris purus lobortis eros, vel viverra lacus felis quis dolor. Quisque condimentum posuere nunc sed iaculis. Nulla efficitur cursus eros non condimentum. Nulla vel ante felis. Morbi vitae est eget tortor ullamcorper blandit.

    Mauris eget tristique elit. Proin consequat odio at mauris suscipit cursus. Fusce justo lacus, ornare at vestibulum id, ultricies vitae lorem. Aliquam maximus ullamcorper nibh, nec varius velit iaculis quis. Aliquam cursus dictum semper. Fusce bibendum, nisl at mattis euismod, ex ante commodo nisi, vitae tincidunt justo nisl non quam. Nunc ultrices accumsan aliquam. Duis rutrum, nunc vitae rhoncus fermentum, nisl quam dignissim dolor, vel varius augue lacus at turpis. Nam lacus nisi, tristique eu libero vel, vulputate dapibus metus. Suspendisse potenti. Mauris ullamcorper lobortis vestibulum. Aliquam eget faucibus velit. Suspendisse potenti. Ut eget tortor ac ex lacinia lobortis eu sit amet lacus.

    Fusce id euismod ipsum. Sed erat dolor, pharetra quis odio eu, iaculis rutrum dolor. Cras vulputate pharetra vulputate. Duis interdum ipsum non elit rutrum, in commodo nunc hendrerit. Ut magna sapien, eleifend nec massa ut, commodo suscipit neque. Nunc in laoreet tortor. Sed hendrerit placerat velit, ut cursus ligula vehicula ut. Nulla sodales auctor magna, non mollis tellus efficitur a. Sed sed metus ut velit aliquet gravida vitae a dui. Donec purus arcu, consectetur scelerisque ipsum in, volutpat rutrum erat. Mauris vehicula ex in elit euismod, sed dignissim elit semper. Ut rutrum feugiat tellus, auctor dictum velit congue vel. Aenean quis risus dolor. Phasellus non elit auctor, eleifend felis gravida, finibus nisl. Duis fermentum enim a eros gravida, vitae posuere dui vehicula.

    Donec ut faucibus leo, in venenatis velit. Nulla ut lacus sit amet leo semper luctus. Sed quis turpis interdum, bibendum libero quis, vulputate purus. Donec mi dolor, viverra vitae dictum facilisis, accumsan a ante. Sed mollis cursus euismod. In scelerisque maximus posuere. Proin gravida metus in lorem tincidunt, nec commodo augue sagittis. Interdum et malesuada fames ac ante ipsum primis in faucibus. Etiam malesuada id erat quis accumsan. Morbi sed euismod justo. Mauris elementum ante nec elit porta, sit amet placerat turpis finibus. Vivamus eu lacinia sem, et faucibus nisi. Vestibulum ultricies eleifend massa. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Aenean tincidunt mi enim, at consequat leo varius sed.

    Nam accumsan bibendum arcu in consequat. Morbi et nulla orci. Proin id interdum eros. Duis pharetra, neque ac iaculis volutpat, nunc metus rhoncus nunc, ac pharetra mi turpis id urna. Quisque nulla orci, pharetra quis urna in, dictum rhoncus nisl. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Praesent et dignissim lectus. Mauris consectetur, justo id rutrum maximus, felis sapien lacinia massa, in dignissim turpis dolor eu nulla. Curabitur scelerisque tellus ultrices venenatis ultrices. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.

    Ut venenatis sodales ex, at euismod leo tempor eu. Cras mattis arcu sit amet dictum tristique. Integer imperdiet sem quis tortor varius, vel sagittis arcu efficitur. Vestibulum tincidunt purus orci, et malesuada dolor tempus id. Cras ultrices dolor vel lectus faucibus imperdiet. Curabitur ullamcorper efficitur lectus ut consequat. Nulla semper lacus a mauris auctor, ac feugiat diam sollicitudin. Sed sed augue luctus, semper dui et, congue est. Nulla placerat, metus et porttitor condimentum, odio augue volutpat sapien, faucibus facilisis neque arcu tempor metus. Nam nec imperdiet mi, sed tincidunt felis. Integer eleifend arcu id mollis faucibus. Phasellus quis nibh sagittis, euismod justo nec, semper diam. Pellentesque accumsan imperdiet massa feugiat suscipit. Praesent suscipit arcu sit amet purus auctor, euismod blandit sem vehicula. Maecenas egestas viverra lobortis. Phasellus vel vestibulum diam.

    Morbi nec dolor faucibus enim fringilla rutrum ut sed arcu. Pellentesque a purus orci. Suspendisse porta libero eu mi pharetra, ut maximus ipsum vulputate. In non urna in ligula posuere porta et at enim. Pellentesque rhoncus metus ac est fringilla, molestie suscipit lacus malesuada. Phasellus sagittis sit amet ante ultrices vulputate. Fusce nec nulla imperdiet, auctor risus quis, ultricies lorem. Maecenas ornare sodales sollicitudin. Integer eleifend, risus a fringilla placerat, elit nulla accumsan augue, id vulputate urna orci a ligula. Integer vestibulum ultricies dolor, quis scelerisque ante porttitor eu. Sed vel neque est.

    Nulla sit amet tellus a tortor hendrerit dapibus a vel turpis. Morbi condimentum iaculis porttitor. Aenean semper vehicula sem, nec convallis lacus sodales ut. Suspendisse maximus mi non nisl tempor ultrices. Sed suscipit porttitor lectus, vitae laoreet nunc efficitur at. Pellentesque id venenatis mauris, vel varius justo. Maecenas in nulla id massa convallis dapibus. Donec sodales metus in urna venenatis, in ultricies augue iaculis.

    Nam egestas porta diam, non tincidunt erat aliquet id. Nulla auctor tellus vulputate ligula ultricies, quis rhoncus orci placerat. Nullam nec sagittis odio. Suspendisse commodo quam non dolor porta, a vestibulum velit porttitor. Maecenas tempus arcu sed massa dignissim, eu iaculis ante tristique. Aliquam sagittis libero quis libero iaculis consectetur. Sed id elit vel diam lobortis egestas. Vivamus vel pretium ligula. Sed tristique, tortor vehicula interdum malesuada, arcu ligula varius lacus, eu tristique erat quam et arcu. Integer cursus ultrices vehicula. Aenean condimentum aliquam nisl ut aliquet. Duis auctor eu ipsum ut porttitor. Morbi luctus, ligula non elementum mattis, felis arcu porttitor purus, eu rutrum felis nisl in elit. Nullam malesuada fringilla erat, nec luctus urna dignissim vitae.

    In quis lorem orci. Donec imperdiet id ligula vel molestie. Maecenas maximus rutrum neque, pretium mollis orci accumsan ut. Nunc eu dolor sagittis, blandit magna et, consectetur ex. Nulla magna lacus, efficitur quis cursus nec, fringilla vitae turpis. Donec vel diam non justo maximus pharetra sit amet pretium odio. Nam in nunc faucibus, rhoncus ipsum rutrum, aliquet nibh. Donec et nunc sit amet enim iaculis dictum. Vestibulum tortor lorem, tempor sed purus et, porta varius lacus. Praesent et urna velit. Cras volutpat lectus diam, sit amet laoreet orci imperdiet dignissim. Sed ullamcorper cursus magna, nec pharetra lectus viverra quis. Sed blandit, sem at accumsan semper, orci lectus semper quam, et pellentesque lectus felis ut turpis. Maecenas laoreet aliquet hendrerit. Sed pulvinar, justo quis consequat tristique, erat leo vehicula turpis, vitae ultricies libero metus vitae felis.

    Suspendisse convallis quis tortor in consequat. Vestibulum sagittis commodo magna rhoncus vehicula. Donec sit amet placerat eros. Sed mattis felis et velit sodales aliquam eu vel nulla. Praesent at finibus erat. Vivamus vel mollis ante. Sed id ultricies ante, non luctus est. Ut pellentesque et purus nec tempus. In auctor, risus vitae malesuada auctor, libero nisl vestibulum justo, in facilisis nisi lorem sit amet augue. Vestibulum hendrerit turpis tincidunt urna hendrerit rutrum. Vestibulum nec libero libero. Morbi porttitor placerat neque quis vestibulum. Cras nec purus at tellus semper laoreet. Nulla tincidunt ornare nisl, id consequat felis commodo eget. Suspendisse placerat ornare mauris, vitae finibus odio finibus eu. Donec tristique velit in aliquam faucibus.

    Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam tristique luctus vestibulum. Cras placerat consequat nibh lobortis lobortis. Praesent euismod varius maximus. Suspendisse finibus felis in massa condimentum elementum vel sed sem. Donec sagittis turpis sed justo egestas, vel tincidunt nisl dapibus. Phasellus imperdiet risus vitae velit hendrerit, at euismod elit aliquet. Nam ut iaculis leo, eget dignissim velit. Duis venenatis condimentum massa, eu mollis ligula tincidunt eu. Nam mattis tempor justo sed rutrum. Maecenas et commodo dui. Mauris fringilla magna erat, id euismod risus faucibus consequat. Pellentesque dapibus eros a commodo condimentum.

    Vivamus hendrerit tincidunt quam, vel tempor magna tristique gravida. Vivamus laoreet arcu lacinia, mattis tellus vel, auctor urna. Vivamus cursus lectus tellus, quis maximus tortor pharetra sed. Donec finibus lectus non sapien porta, nec scelerisque felis finibus. Nulla facilisi. Etiam commodo lobortis vulputate. Praesent auctor vitae odio eu dapibus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Sed ut arcu a quam eleifend semper eget ut felis. Maecenas eleifend tellus nisi, eget dictum nisi accumsan eu. Aliquam ex libero, pharetra quis ex non, finibus sollicitudin arcu. Etiam venenatis aliquam est, a ullamcorper dui maximus sit amet. Proin a elementum dui, finibus semper nulla. Mauris vel vehicula sem, in fermentum augue. In mollis varius leo non semper. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

    Vivamus dignissim lacus vitae arcu convallis, a ultricies augue suscipit. Nunc ut odio elementum, pretium ante ut, fermentum purus. Etiam venenatis pellentesque libero. Morbi ac malesuada leo. Curabitur semper odio eu magna pulvinar suscipit. Curabitur eros sapien, efficitur ut libero et, aliquam dignissim turpis. Nam et enim ante. Morbi facilisis mauris tempus, bibendum justo vel, blandit nunc.

    Sed facilisis, libero quis sollicitudin faucibus, justo sem feugiat nibh, a auctor nisi ligula in lorem. Nam sit amet orci nec nisi fermentum interdum eget in erat. In tempus, orci feugiat laoreet lobortis, odio augue convallis mi, a volutpat lacus dui ut urna. Etiam ultricies sapien turpis, at pretium turpis pharetra eget. Suspendisse nec ligula enim. Vivamus quis eros est. Sed efficitur, massa nec tincidunt viverra, lorem diam euismod neque, eu iaculis mi ante sed leo. Sed tempor tempor massa ac pharetra.

    Sed luctus leo vel dolor dapibus, ac auctor libero interdum. Aliquam sollicitudin fringilla massa non euismod. Curabitur molestie eros bibendum mattis imperdiet. Quisque ut pulvinar ante. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Sed cursus sem gravida mi porta faucibus. Duis commodo augue nec ante sagittis, eu vestibulum risus maximus. Nunc in imperdiet nunc, ut finibus est.

    Duis convallis dui ut sem pellentesque interdum. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nunc efficitur sapien ut nisi luctus egestas. Vivamus non arcu laoreet, convallis tellus eget, semper velit. Pellentesque tempor tempor metus, vel imperdiet dui tempor vitae. Nulla facilisi. Ut consectetur venenatis justo, non blandit erat dignissim ut. Aliquam ac elementum magna, ac laoreet ex. Etiam gravida nec quam nec porta.

    Donec euismod sagittis mauris nec congue. Nulla lobortis erat ac ex fermentum blandit. Duis rutrum dui ut eros sollicitudin blandit. Proin malesuada varius lacus efficitur scelerisque. In hac habitasse platea dictumst. Nulla mollis purus at ipsum vehicula semper. Integer sit amet arcu tortor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse efficitur magna commodo faucibus lobortis. Nam fermentum sem sit amet lobortis venenatis. Nullam eleifend eu justo in venenatis. Quisque ac sagittis magna.

    Ut urna neque, auctor non nisl nec, ultrices vehicula quam. Maecenas ut tempor ante. Pellentesque at imperdiet lorem. Proin ac molestie orci, vel porttitor tellus. Proin id ligula consequat, cursus mauris ut, faucibus tellus. Sed bibendum molestie finibus. Cras aliquam fermentum sapien. Phasellus auctor dignissim bibendum. Mauris placerat, augue ut euismod iaculis, elit nibh pretium odio, et viverra sem nulla a nibh. Sed lobortis dui ut purus consectetur elementum. Curabitur maximus facilisis dui vel rhoncus. Sed quis nibh massa. Suspendisse suscipit tristique auctor. Aenean hendrerit venenatis nisl ac dictum. Nullam sit amet felis et tellus congue finibus sed id augue.

    Phasellus vitae dictum leo. Mauris dapibus consequat est id maximus. Proin maximus sed mauris a venenatis. Integer fringilla enim metus, vel efficitur sem convallis id. Aenean commodo, nisl efficitur scelerisque lacinia, orci urna condimentum velit, et tempor ipsum diam condimentum justo. Etiam non velit justo. Aliquam pharetra elit et hendrerit hendrerit. Cras pretium at lorem ut venenatis. Ut ultrices, tortor ac euismod rutrum, turpis odio fermentum purus, facilisis tristique lacus est quis justo. Vivamus posuere, neque vitae malesuada volutpat, velit justo sodales magna, interdum egestas turpis libero non dui. Sed at augue laoreet, suscipit est id, malesuada mi.

    Duis lacinia quam eget orci euismod convallis. Nulla erat turpis, viverra et ornare a, sagittis vel sapien. Cras iaculis, dolor semper sodales commodo, eros dui dictum velit, sed porttitor felis quam in lectus. Fusce id ligula non justo venenatis hendrerit efficitur in urna. Cras ex mauris, convallis nec eros quis, commodo tincidunt eros. Etiam hendrerit, augue sit amet ultricies aliquam, tortor felis dapibus risus, sit amet ultricies nisl lorem et lectus. Duis feugiat ultrices est, eu faucibus odio volutpat in. Donec enim risus, molestie a purus vel, consequat dictum neque. Mauris felis nibh, hendrerit at molestie eu, laoreet non nisl. Donec odio eros, faucibus a convallis at, ornare et tortor. Nulla id lacinia mi.

    Mauris et lectus erat. Morbi iaculis quam vitae arcu fermentum, in rutrum metus gravida. Praesent cursus dapibus mi, nec mollis orci aliquam ac. Praesent elementum auctor tortor, in cursus arcu porttitor vitae. Donec ac augue nec elit elementum suscipit eu lacinia erat. Cras varius viverra felis, vitae tempor eros elementum ac. In porttitor, nibh a pulvinar congue, lacus dui consectetur risus, eget feugiat felis lectus dignissim nulla. Nunc egestas est eu mi ultricies, eget efficitur tortor sodales. Mauris nec ornare velit, ac molestie orci. Phasellus sit amet mi eget nisi pretium viverra. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Integer nec ligula non elit vulputate aliquam. Maecenas blandit sem et suscipit pharetra.

    In hac habitasse platea dictumst. Suspendisse pulvinar vitae arcu et varius. Nulla eget urna ullamcorper, rutrum lectus sed, accumsan magna. Maecenas varius justo ut metus tristique sagittis. Fusce tempor arcu sed fringilla tincidunt. Fusce pharetra nec dui sit amet rutrum. Donec id mollis nisi.

    Donec rutrum, nisl a auctor tincidunt, nisi sapien interdum turpis, a imperdiet odio diam fermentum tortor. Cras vel augue euismod, pulvinar nunc in, volutpat odio. Quisque ac nulla leo. Maecenas sit amet purus vitae mauris accumsan semper. Nullam accumsan purus a aliquet volutpat. Sed sollicitudin nisi in fringilla faucibus. Morbi eget lacus a ipsum vulputate dapibus in sit amet est. Duis sed arcu nunc. Donec condimentum enim ac enim volutpat convallis.

    Aliquam vel rhoncus justo. Nunc id est dictum tellus aliquet sagittis quis eu nisl. Donec vestibulum nisl eu nunc tincidunt, vitae suscipit urna suscipit. Ut facilisis, risus vel mattis ultrices, nulla ligula mattis risus, in molestie nibh justo in arcu. Donec ut vulputate lectus. Donec dignissim odio id vestibulum semper. Praesent tincidunt, lacus vel porta condimentum, orci metus porttitor ante, eu dignissim velit libero vel odio. Donec lectus est, congue in luctus eget, molestie eu nunc. Fusce sit amet orci eros. Curabitur elementum odio id risus condimentum euismod at eu est. Curabitur luctus, mauris et venenatis tincidunt, mi odio sollicitudin turpis, quis elementum purus purus eu sapien. Aliquam blandit fermentum dolor. Nulla ullamcorper faucibus purus, nec bibendum turpis tincidunt nec. Fusce malesuada, orci at venenatis semper, ante lorem sagittis sapien, quis pellentesque felis lectus at leo.

    Mauris posuere suscipit leo, et ultrices nisi. In hac habitasse platea dictumst. Mauris lorem orci, porttitor eu purus nec, vestibulum feugiat est. Praesent tincidunt dapibus dictum. Sed congue libero eros, in euismod leo scelerisque id. Suspendisse potenti. Aenean ornare venenatis consectetur. Morbi feugiat diam non ligula gravida pellentesque. Sed velit est, convallis eget mauris at, viverra blandit augue. Phasellus pharetra et metus a auctor. Integer feugiat tellus eu ligula pretium, nec mollis urna sodales. Sed lacinia ornare diam at tincidunt. Nam laoreet sapien quis feugiat faucibus. Maecenas imperdiet quis nisi sed mattis. Nunc at vestibulum ante. Nam a odio ac elit finibus blandit.

    Nunc aliquet pellentesque dictum. Donec placerat, ex vel dapibus vulputate, lectus leo faucibus ligula, eu commodo felis nibh nec neque. Praesent vitae arcu mattis, pretium erat facilisis, blandit augue. Cras quis velit pretium, faucibus nisl id, ullamcorper mauris. Ut dapibus lorem in nisi fringilla consectetur. Pellentesque consectetur leo at dolor congue vulputate rutrum vel augue. Donec fringilla, nibh non sagittis sagittis, diam elit eleifend massa, ut accumsan massa mi non lectus. In enim ipsum, laoreet id convallis eu, hendrerit ut odio. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Morbi faucibus pulvinar augue, eget sodales libero consectetur sit amet. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Quisque efficitur, tortor in efficitur cursus, diam nisi vestibulum dolor, vel ultricies mauris nunc in justo. Nullam dignissim mauris ut sodales porta.

    Nullam ornare ac metus at condimentum. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Quisque a rutrum ligula. Vivamus at purus lacinia, feugiat felis a, dapibus mi. Praesent quis enim iaculis, accumsan massa at, imperdiet dolor. Curabitur sollicitudin erat a nulla hendrerit luctus vitae vulputate ex. Nulla rutrum vitae enim a sodales. In nisi purus, tincidunt laoreet tellus vestibulum, elementum ultrices augue. Maecenas aliquam sapien eros, vel maximus orci tincidunt id. Cras est lorem, interdum pulvinar congue quis, hendrerit sed nulla.

    Nam at gravida odio, in pharetra turpis. Mauris et magna ligula. Proin elementum cursus viverra. Donec id aliquet elit. In pretium posuere malesuada. Nunc non arcu vitae mauris tempor facilisis. Sed id malesuada eros. Suspendisse ultrices, odio sit amet mollis euismod, velit justo luctus massa, et euismod lectus orci eu tellus. Morbi mollis elit in orci egestas luctus. Phasellus tincidunt efficitur ligula ut cursus. Ut sed justo at felis porttitor vehicula. Suspendisse eu facilisis nunc. Pellentesque auctor sodales ex, vitae auctor nisl vestibulum id. Nam pretium sed sem at dapibus. Mauris et ultricies tellus.

    Donec facilisis vehicula felis quis finibus. Maecenas auctor interdum mauris hendrerit volutpat. Integer non vulputate tellus, sit amet tristique libero. Duis vestibulum, lacus sed fringilla accumsan, ipsum risus lacinia nisl, eget scelerisque quam enim a nunc. Sed varius orci id turpis porttitor rutrum. Ut a fermentum turpis, ac hendrerit augue. Integer convallis mauris dui, eu scelerisque quam aliquam a. Aenean maximus quam nibh, a ultricies justo egestas vitae. Duis at feugiat arcu. Donec pellentesque vehicula libero et blandit. Donec tempor felis a metus convallis vulputate. Quisque nec porttitor neque. In et lobortis risus. Integer suscipit purus velit, eu euismod nisl elementum vel. Suspendisse efficitur magna ac massa fermentum ornare. Aliquam velit ipsum, molestie a ante ut, feugiat ullamcorper orci.

    In nec suscipit augue. Mauris sagittis odio eu est varius, in hendrerit velit egestas. Donec a eros non purus mollis molestie quis quis nisl. Curabitur eros diam, scelerisque nec purus pulvinar, consectetur ornare metus. Nam nec leo et turpis tincidunt convallis eu non dolor. Maecenas elementum leo molestie, maximus nunc a, laoreet lorem. Donec varius nisl dictum nibh molestie laoreet. Integer non felis ac ex ornare ultricies. Curabitur dolor leo, hendrerit et semper a, sollicitudin sed magna. Mauris sit amet odio at odio condimentum volutpat quis a erat.

    Morbi sodales convallis pretium. Nullam pellentesque et purus sed tristique. Nam bibendum tellus consequat, rhoncus ex at, efficitur risus. Sed convallis est eros, et mattis elit pulvinar eu. Nullam id velit vel leo volutpat suscipit sed in erat. Nam fringilla, neque vel rhoncus sagittis, mauris sapien lacinia enim, id blandit nisi sapien a lacus. Morbi mollis lectus quis dui hendrerit, a rhoncus odio lobortis. Aenean eget magna sed risus ultricies finibus.

    Duis vel libero dapibus, ullamcorper lorem ut, molestie metus. Sed quis efficitur libero, a consequat mi. Curabitur accumsan libero vel dignissim scelerisque. Praesent laoreet leo non nisi fringilla, id lobortis urna luctus. Nulla vel augue a nulla pellentesque pharetra cursus vel nisl. Proin in augue vel leo molestie tincidunt. Praesent tempus sem ut diam pretium, nec imperdiet libero ullamcorper. Cras pretium magna eu arcu accumsan, ut dignissim urna tempus. Proin non placerat nisl. Morbi ultricies, libero sit amet molestie suscipit, magna nisi consequat nisl, sit amet vehicula mauris arcu quis nisl. Vestibulum tempor dapibus feugiat.

    Nullam sed maximus lectus. Sed vel maximus eros. Vestibulum nec quam at dui fringilla condimentum. Phasellus felis tortor, dignissim vel pharetra et, convallis a leo. Cras fringilla viverra urna, nec hendrerit nibh tincidunt pulvinar. Sed est purus, egestas at malesuada et, pharetra ac magna. Donec vehicula, eros nec commodo ullamcorper, risus mi facilisis dolor, at pharetra dolor nulla sed mauris. Nullam non maximus felis. Sed sodales scelerisque ornare. Sed non mollis nibh. Proin pulvinar nisi id ante mattis interdum at et sapien. Phasellus congue tincidunt erat in varius. Duis sit amet auctor risus.

    Nullam condimentum tellus urna, pharetra rhoncus nisi interdum quis. Nulla vitae massa lectus. Sed venenatis nulla sit amet justo interdum, id porta dolor dictum. Ut nisl augue, posuere ac venenatis sed, facilisis vel nisl. Donec odio leo, hendrerit nec hendrerit eu, gravida non tortor. Duis nec commodo justo. Sed faucibus luctus dui sed vehicula. Phasellus placerat dignissim feugiat. Quisque dui metus, scelerisque non aliquam at, varius a sapien. Donec tristique erat sem, ac iaculis tellus pulvinar non. Nulla sit amet quam id magna volutpat commodo. Pellentesque magna massa, suscipit a maximus vel, tincidunt quis purus. Sed non tortor nunc. Vestibulum vel diam interdum, eleifend metus quis, accumsan augue. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae;

    Mauris ac auctor enim. Cras id nulla tempus, efficitur turpis egestas, varius massa. Fusce mattis, orci in sagittis auctor, orci elit luctus leo, gravida viverra ex erat a justo. Praesent mi ipsum, ultricies quis erat ut, interdum pretium risus. Sed nec tellus iaculis, suscipit elit id, molestie augue. Sed vel orci ut elit rhoncus fringilla. Vestibulum in ante non justo hendrerit interdum sed id urna. Nulla porttitor, orci at feugiat porta, felis risus facilisis justo, eu hendrerit lacus ex ac turpis. Mauris sit amet metus ut eros vehicula dictum.

    Ut quis magna felis. Etiam finibus tortor ligula, consequat ultricies mi laoreet eu. Donec rhoncus ligula non tincidunt imperdiet. Ut viverra imperdiet felis eu bibendum. Sed ut finibus nisi. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aenean consequat mauris eget enim pellentesque, nec convallis nisl dapibus. Proin ac gravida justo, quis sollicitudin metus. Pellentesque condimentum ipsum nec libero pharetra interdum. Proin malesuada, urna vitae convallis cursus, lectus ex tincidunt eros, non viverra mi nisi sagittis enim. Vivamus consequat pretium augue, sit amet eleifend sem congue eu. Sed luctus ultrices turpis, sit amet varius dolor condimentum quis. Nullam commodo, purus nec suscipit feugiat, lectus metus gravida ante, non tincidunt nisl augue nec felis.

    Proin id leo quam. Duis eget ornare turpis. Vivamus fringilla accumsan orci, pharetra semper orci gravida at. Maecenas blandit nunc a ipsum blandit, a commodo lectus pulvinar. Quisque ut felis id lectus eleifend commodo. Vestibulum a velit ut arcu sagittis vulputate. Integer convallis, est sed maximus dapibus, eros dolor lobortis nisi, vitae sollicitudin lectus metus eget lacus. Etiam ut sapien orci. Maecenas aliquam tortor lacus, eu commodo nulla finibus at. Aliquam cursus turpis magna, nec blandit nunc eleifend a. In laoreet ornare dolor, eget sollicitudin felis congue at. Vestibulum ullamcorper elit tellus, vitae blandit diam rutrum quis. Mauris tristique eros ac sem porttitor, eget condimentum enim tristique.

    Mauris condimentum dignissim condimentum. Sed vel urna dapibus, condimentum felis sed, gravida urna. Proin condimentum, felis vel commodo finibus, justo dolor ornare velit, ultrices consectetur nisi erat vitae lectus. Vivamus sit amet libero porttitor erat egestas mollis vitae a lacus. Nullam in posuere nunc. Aenean blandit tortor ac faucibus fringilla. Integer vel nisi in nulla eleifend imperdiet non eget lectus. Nullam ut fringilla mi, vitae imperdiet turpis. Aliquam ut est nisi. Praesent elementum est lacus, eget imperdiet purus pulvinar ut. Cras eget nunc commodo, viverra purus at, commodo leo. Sed vitae augue a metus hendrerit viverra mollis in metus. Vestibulum eget rhoncus justo, sed maximus tellus.

    Duis vitae augue vitae quam sollicitudin consequat. Donec ac efficitur metus. Maecenas feugiat at tellus a efficitur. In hac habitasse platea dictumst. Praesent a mollis orci. Donec id posuere nibh, consequat malesuada purus. Quisque sodales dolor sed ex vestibulum vulputate.
    """
    return text.split(". ")
