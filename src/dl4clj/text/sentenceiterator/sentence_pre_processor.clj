(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/SentencePreProcessor.html"}
  dl4clj.text.sentenceiterator.sentence-pre-processor
  (:import [org.deeplearning4j.text.sentenceiterator SentencePreProcessor]))

(defn pre-process
  [^SentencePreProcessor p ^String s]
  (.preProcess p s))

(deftype CustomSentencePreprocessor [preprocessor]
  SentencePreProcessor
  (preProcess [this s]
    (preprocessor s)))


(defn sentence-pre-processor [preprocessor]
  (CustomSentencePreprocessor. preprocessor))

(comment

  (def p (sentence-pre-processor clojure.string/lower-case))

  (pre-process p "Foo Bar !")

  )
