(ns ^{:doc "see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/paragraphvectors/tools/MeansBuilder.java"}
  dl4clj.examples.paragraphvectors.tools.means-builder
  (:require [dl4clj.models.embeddings.inmemory.in-memory-lookup-table :refer (get-vocab)]
            [dl4clj.utils :refer (indexed)]
            [dl4clj.models.word2vec.wordstore.vocab-cache :refer (contains-word)]
            [dl4clj.text.tokenization.tokenizerfactory.tokenizer-factory :as tf]
            [dl4clj.text.tokenization.tokenizerfactory.tokenizer :refer (get-tokens)]
            [dl4clj.text.documentiterator.labelled-document :refer (get-content)]
            [nd4clj.linalg.factory.nd4j :as ndf]
            [dl4clj.models.embeddings.weight-lookup-table :as lt]
            [nd4clj.linalg.api.ndarray.indarray :refer (put-row mean)]))

(defn document-as-vector
  "Computes and returns centroid (mean vector) for document (an INDArray)."
  [lookup-table tokenizer-factory vocab-cache document]
  (let [document-as-tokens (filter (partial contains-word vocab-cache)
                                   (get-tokens (tf/create-from-string tokenizer-factory (get-content document))))
        all-words (ndf/create-from-shape (count document-as-tokens)
                                         (lt/layer-size lookup-table))]
    (doseq [[word idx] (indexed document-as-tokens)]
      (put-row all-words idx (lt/vector lookup-table word)))
    (mean all-words 0)))

