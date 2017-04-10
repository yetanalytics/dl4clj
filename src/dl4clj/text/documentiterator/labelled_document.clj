#_(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/documentiterator/LabelledDocument.html"}
  dl4clj.text.documentiterator.labelled-document
  (:import [org.deeplearning4j.text.documentiterator LabelledDocument]))

#_(defn set-content [^LabelledDocument d content]
  (.setContent d content))

#_(defn set-label [^LabelledDocument d label]
  (.setLabel d label))

#_(defn get-content [^LabelledDocument d]
  (.getContent d))

#_(defn get-label [^LabelledDocument d]
  (.getLabel d))

#_(defn labelled-document
  ([]
   (LabelledDocument.))
  ([content label]
   (let [d (LabelledDocument.)]
     (set-content d content)
     (set-label d label)
     d)))

(comment

  (def d (labelled-document))
  (set-content d "foo bar")
  (get-content d)

)
