(ns ^{:doc "see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/paragraphvectors/ParagraphVectorsClassifierExample.java
and https://cs.stanford.edu/~quocle/paragraph_vector.pdf"}
  dl4clj.examples.paragraphvectors.paragraph-vectors-classifier-example
  (:require [dl4clj.text.documentiterator.label-aware-iterator :refer (documents)]
            [dl4clj.examples.paragraphvectors.tools.file-label-aware-iterator :refer (file-label-aware-iterator)]
            [dl4clj.text.tokenization.tokenizerfactory.default-tokenizer-factory :refer (default-tokenizer-factory)]
            [dl4clj.text.tokenization.tokenizerfactory.tokenizer-factory :refer (set-token-pre-processor)]
            [dl4clj.text.tokenization.tokenizer.preprocessor.common-preprocessor :refer (common-preprocessor)]
            [dl4clj.models.paragraphvectors.paragraph-vectors :refer (paragraph-vectors)]
            [dl4clj.models.sequencevectors.sequence-vectors :refer (fit)]
            [dl4clj.models.embeddings.inmemory.in-memory-lookup-table :refer (get-vocab)]
            [dl4clj.models.embeddings.wordvectors.word-vectors :refer (lookup-table)]
            [dl4clj.examples.paragraphvectors.tools.means-builder :refer :all]
            [dl4clj.text.documentiterator.labels-source :refer (get-labels)]
            [dl4clj.examples.paragraphvectors.tools.label-seeker :refer (get-scores)]
            [dl4clj.text.documentiterator.labelled-document :refer (get-label)]))

;; build an iterator for our dataset
(def iterator (file-label-aware-iterator (clojure.java.io/resource "paravec/labeled")))

(def t (default-tokenizer-factory))
(set-token-pre-processor t (common-preprocessor))


;; ParagraphVectors training configuration
(def paragraphvectors (paragraph-vectors {:learning-rate 0.025
                                          :min-learning-rate 0.001
                                          :batch-size 1000
                                          :epochs 20
                                          :iterate iterator
                                          :train-word-vectors true
                                          :tokenizer-factory t}))

;; Start model training
(time (fit paragraphvectors))
"Elapsed time: 77194.801 msecs"

;; At this point we assume we have a model and we can classify unlabeled documents.
(def unlabeled-iterator (file-label-aware-iterator (clojure.java.io/resource "paravec/unlabeled")))
 
(def lt (lookup-table paragraphvectors))
(def v (get-vocab lt))
(def labels (get-labels (:labels-source iterator)))

(doseq [document (documents unlabeled-iterator)]
  (let [scores (get-scores labels lt (document-as-vector lt t v document))]
    (println (str "Document '" (get-label document) "' falls into the following categories: "))
    (println scores)))

;; => Document 'finance' falls into the following categories: 
;;    {science 0.018949185163223244, health -0.11507322176755937, finance 0.40230107446722246}
;;    Document 'health' falls into the following categories: 
;;    {science -0.022660476948528845, health 0.5559929583559113, finance -0.3899286864778339}
