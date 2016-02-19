(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/wordvectors/WordVectorsImpl.html"}
  dl4clj.models.embeddings.wordvectors.word-vectors-impl
  (:import [org.deeplearning4j.models.embeddings.wordvectors WordVectorsImpl]))

(defn word-vectors-impl []
  (WordVectorsImpl.))
