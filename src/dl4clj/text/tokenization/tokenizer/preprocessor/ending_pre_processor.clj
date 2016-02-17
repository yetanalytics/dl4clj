(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/preprocessor/EndingPreProcessor.html"}
  dl4clj.text.tokenization.tokenizer.preprocessor.ending-pre-processor
  (:require [dl4clj.text.tokenization.tokenizer.token-pre-process :refer :all])
  (:import [org.deeplearning4j.text.tokenization.tokenizer.preprocessor EndingPreProcessor]))

(defn ending-pre-processor []
  (EndingPreProcessor.))

(defmethod pre-process EndingPreProcessor [^EndingPreProcessor pp ^String token]
  (.preProcess pp token))
