(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/TokenPreProcess.html"}
  dl4clj.text.tokenization.tokenizer.token-pre-process
  (:require [dl4clj.text.tokenization.tokenizer.token-pre-process :refer :all])
  (:import [org.deeplearning4j.text.tokenization.tokenizer TokenPreProcess]))

(defmulti pre-process
  "Pre process a token"
  (fn [x token] (type x)))

(deftype CustomTokenPreProcessor [preprocessor]
  TokenPreProcess
  (preProcess [this x] (preprocessor x)))

(defmethod pre-process CustomTokenPreProcessor [^CustomTokenPreProcessor this x]
  (.preProcess this x))

(defn token-pre-process [processor]
  (CustomTokenPreProcessor. processor))
