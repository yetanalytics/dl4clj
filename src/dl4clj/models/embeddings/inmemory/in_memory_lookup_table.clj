(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/inmemory/InMemoryLookupTable.html"}
  dl4clj.models.embeddings.inmemory.in-memory-lookup-table
  (:import [org.deeplearning4j.models.embeddings.inmemory InMemoryLookupTable]
           [org.deeplearning4j.plot Tsne]))

(defn in-memory-lookup-table
  ([]
   (InMemoryLookupTable.))
  ([vocab-cache vector-length use-ada-grad? learning-rate rng negative]
   (InMemoryLookupTable. vocab-cache (int vector-length) (boolean use-ada-grad?) (double learning-rate) rng (double negative))))
  
(defn get-vocab [^InMemoryLookupTable this]
  (.getVocab this))
