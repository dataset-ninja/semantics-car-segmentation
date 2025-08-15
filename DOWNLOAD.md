Dataset **Car Segmentation** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMjUwNF9DYXIgU2VnbWVudGF0aW9uL2Nhci1zZWdtZW50YXRpb24tRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAiQlIxa0szWHpzREFEdDdzNVZSOFVHY01uOHlNblhnTDhjZy9WR3VMcHZTQT0ifQ==?response-content-disposition=attachment%3B%20filename%3D%22car-segmentation-DatasetNinja.tar%22)

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

