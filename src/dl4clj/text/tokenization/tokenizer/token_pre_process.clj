(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/TokenPreProcess.html"}
  dl4clj.text.tokenization.tokenizer.token-pre-process
  (:require [dl4clj.text.tokenization.tokenizer.token-pre-process :refer :all])
  (:import [org.deeplearning4j.text.tokenization.tokenizer TokenPreProcess]))

(defn pre-process
  "Pre process a token"
  [^TokenPreProcess x ^String token]
  (.preProcess x token))

(deftype CustomTokenPreProcessor [preprocessor]
  TokenPreProcess
  (preProcess [this x] (preprocessor x)))

(defn token-pre-process [processor]
  (CustomTokenPreProcessor. processor))
