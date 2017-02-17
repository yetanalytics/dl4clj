(ns ^{:doc ""}
  dl4clj.text.tokenization.tokenizerfactory.default-tokenizer-factory
  (:import [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory]
           [org.deeplearning4j.text.tokenization.tokenizer TokenPreProcess]))

(defn default-tokenizer-factory []
  (DefaultTokenizerFactory.))
