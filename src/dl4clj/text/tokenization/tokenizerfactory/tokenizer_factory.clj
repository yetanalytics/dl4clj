(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizerfactory/TokenizerFactory.html"}
  dl4clj.text.tokenization.tokenizerfactory.tokenizer-factory
  (:import [org.deeplearning4j.text.tokenization.tokenizerfactory TokenizerFactory]))

(defmulti create 
  "Create a tokenizer based on an input stream"
  (fn [x is] (type x)))

(defmulti create-from-string 
  "Create a tokenizer from string"
  (fn [x s] (type x)))

(defmulti set-token-pre-processor 
  "Sets a token pre processor to be used with every tokenizer" 
  (fn [x pp] (type x)))
