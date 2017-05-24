# dl4clj

Port of [deeplearning4j](https://github.com/deeplearning4j/) to clojure

## Usage

Under construction. For now, have a look at the [examples](https://github.com/engagor/dl4clj/tree/master/src/dl4clj/examples) to get started.  You can also look at the tests (not complete)

## Artifacts

dl4clj artifacts are released to Clojars.

If using Maven add the following repository definition to your pom.xml:

```
<repository>
  <id>clojars.org</id>
  <url>http://clojars.org/repo</url>
</repository>
```

## Latest release

With Leiningen:

```
[engagor/clj-vw "0.0.1"]

```

With Maven:

```
<dependency>
  <groupId>engagor</groupId>
  <artifactId>dl4clj</artifactId>
  <version>0.0.1</version>
</dependency>
```

## Features

Not all of these are fully tested and are most likely going to undergo breaking changes

- Clustering
- Datasets
- Early Stopping
- Eval/Evaluation
- Neural Networks DSL
- Optimize

## TODO

Finish tests of currently implemented classes/interfaces

Refactor overall structure of this project
- ensure consistency in stlye (multimethods for heavy lifting and fns for use)
- ensure no cascading config maps
- seperation of interfaces from the classes/namespaces that implement/use them
- general refinement

Improve examples
- minimal importing
- replace method calls with fn calls

Update doc strings
- content and formatting

Fix built-in dl4j logging

ND4j and Datavec implementations
- some of this has already been done but is no where close to complete

Update release section

Spark and Kafka streaming

## Packages to implement:

Implement ComputationGraphs and the classes which use them
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ComputationGraphConfiguration.GraphBuilder.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ComputationGraphConfiguration.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/graph/package-frame.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/graph/rnn/package-frame.html>

Storage
- <https://deeplearning4j.org/doc/org/deeplearning4j/api/storage/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/api/storage/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/api/storage/listener/RoutingIterationListener.html>

AWS
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/dataset/DataSetLoader.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/ec2/Ec2BoxCreator.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/ec2/provision/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/s3/BaseS3.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/s3/reader/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/s3/uploader/S3Uploader.html>

NLP
- <https://deeplearning4j.org/doc/org/deeplearning4j/bagofwords/vectorizer/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/datasets/vectorizer/Vectorizer.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/iterator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/iterator/provider/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/WeightLookupTable.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/inmemory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/learning/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/learning/impl/elements/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/learning/impl/sequence/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/loader/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/reader/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/reader/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/wordvectors/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/glove/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/glove/count/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/node2vec/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/paragraphvectors/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/enums/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/interfaces/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/iterators/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/listeners/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/sequence/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/serialization/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/transformers/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/transformers/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/transformers/impl/iterables/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/iterator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/wordstore/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/wordstore/inmemory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/annotator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/corpora/sentiwordnet/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/corpora/treeparser/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/corpora/treeparser/transformer/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/documentiterator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/documentiterator/interoperability/DocumentIteratorConverter.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/inputsanitation/InputHomogenization.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/invertedindex/InvertedIndex.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/labels/LabelsProvider.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/movingwindow/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/interoperability/SentenceIteratorConverter.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/labelaware/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/stopwords/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/preprocessor/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/tokenprepreprocessor/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizerfactory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/uima/UimaResource.html>

Kuromoji (https://github.com/atilika/kuromoji)
self-contained and very easy to use Japanese morphological analyzer designed for search
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/buffer/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/compile/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/dict/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/io/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/ipadic/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/ipadic/compile/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/trie/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/util/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/viterbi/package-summary.html>

Keras
- <https://deeplearning4j.org/doc/org/deeplearning4j/keras/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/layers/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/preprocessors/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/trainedmodels/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/trainedmodels/Utils/package-summary.html>

Berkeley
- <https://deeplearning4j.org/doc/org/deeplearning4j/berkeley/package-summary.html>

RecordMetaData
- <https://deeplearning4j.org/doc/org/deeplearning4j/eval/meta/Prediction.html>

MNIST (not sure if necessary)
- <https://deeplearning4j.org/doc/org/deeplearning4j/datasets/mnist/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/datasets/mnist/draw/package-summary.html>


Parallelism
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/factory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/main/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/parameterserver/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/trainer/package-summary.html>

TSNE
- <https://deeplearning4j.org/doc/org/deeplearning4j/plot/package-summary.html>

Utils
- <https://deeplearning4j.org/doc/org/deeplearning4j/util/package-summary.html>

UI
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/activation/PathUpdate.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/api/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/chart/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/chart/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/component/ComponentDiv.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/component/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/decorator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/decorator/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/table/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/table/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/text/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/text/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/flow/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/flow/beans/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/flow/data/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/i18n/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/convolutional/ConvolutionalListenerModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/defaultModule/DefaultModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/flow/FlowListenerModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/histogram/HistogramModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/remote/RemoteReceiverModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/train/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/tsne/TsneModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/nearestneighbors/word2vec/NearestNeighborsQuery.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/play/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/play/misc/FunctionUtil.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/play/staticroutes/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/providers/ObjectMapperProvider.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/renders/PathUpdate.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/standalone/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/api/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/impl/java/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/sbe/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/storage/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/storage/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/storage/mapdb/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/storage/sqlite/J7FileStatsStorage.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/weights/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/weights/beans/CompactModelAndGradient.html>

## License

Copyright © 2016 Engagor

Distributed under the BSD Clause-2 License as distributed in the file LICENSE at the root of this repository.


*Drops the mic*
