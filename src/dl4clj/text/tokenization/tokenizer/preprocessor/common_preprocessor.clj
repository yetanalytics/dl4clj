(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/preprocessor/CommonPreprocessor.html"}
  dl4clj.text.tokenization.tokenizer.preprocessor.common-preprocessor
  (:require [dl4clj.text.tokenization.tokenizer.token-pre-process :refer :all])
  (:import [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]))

(defn common-preprocessor []
  (CommonPreprocessor.))

(defmethod pre-process CommonPreprocessor [^CommonPreprocessor pp ^String token]
  (.preProcess pp token))
