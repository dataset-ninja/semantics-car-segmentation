Dataset **Car Segmentation** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/E/X/0Z/ooceWhxKlUQ7WsRp3kSZ9KbAJgdTfl5Rir0aPlYazYaGQq5dmif8ptYV3LUCUXYOV0hdLAYsssrowDyp5qy7vviuEWtyVnGf2cNZsuO9dTjNXL9HqWigNu9zjUsJ.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Car Segmentation', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

