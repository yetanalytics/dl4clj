(ns ^{:doc ""}
  dl4clj.text.tokenization.tokenizerfactory.default-tokenizer-factory
  (:require [dl4clj.text.tokenization.tokenizerfactory.tokenizer-factory :refer :all])
  (:import [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory]
           [org.deeplearning4j.text.tokenization.tokenizer TokenPreProcess]))

(defn default-tokenizer-factory []
  (DefaultTokenizerFactory.))

(defmethod create DefaultTokenizerFactory [^DefaultTokenizerFactory x ^java.io.InputStream is]
  (.create x is))

(defmethod create-from-string DefaultTokenizerFactory [^DefaultTokenizerFactory x ^java.lang.String s]
  (.create x s))

(defmethod set-token-pre-processor DefaultTokenizerFactory [^DefaultTokenizerFactory x ^TokenPreProcess pp]
  (.setTokenPreProcessor x pp))
