(ns ^{:doc ""}
  dl4clj.examples.example-utils
  (:import [java.io IOException]
           [org.apache.commons.io FileUtils]))

(defn index-map 
  "Utility function to make an map from elements in a collection to indices"
  [col]
  (zipmap col (range)))

(def ^:const +minimal-character-set+
  (let [valid-chars (concat (for [c (range (int \a) (inc (int \z)))] (char c))
                            (for [c (range (int \A) (inc (int \Z)))] (char c))
                            (for [c (range (int \0) (inc (int \9)))] (char c))
                            [\!, \&, \(, \), \?, \-, \\, \", ,, \., \:, \;, \space , \newline, \tab])]
    (into #{} valid-chars)))

(def ^:const +default-character-set+
  (let [valid-chars (concat +minimal-character-set+
                            [\@, \#, \$, \%, \^, \*, \{, \}, \[, \], \/, \+, \_, \\\, \|, \<, \>])]
    (into #{} valid-chars)))

(defn shakespeare-file []
  "Downloads and returns the complete works of Shakespeare as a string.

   5.3MB file in UTF-8 Encoding, ~5.4 million characters
  https://www.gutenberg.org/ebooks/100
"
  (let [url "https://s3.amazonaws.com/dl4j-distribution/pg100.txt"
        temp-dir  (System/getProperty "java.io.tmpdir")
        file-location  (str temp-dir "/Shakespeare.txt")
        f (clojure.java.io/as-file file-location)]
    (when-not (.exists f)
      (do (FileUtils/copyURLToFile (clojure.java.io/as-url url) f)
          (println "File downloaded to " (.getAbsolutePath f))))
    (when-not (.exists f) 
      (throw (IOException. (str "File does not exist: " file-location))))
    f))

(defn ^String shakespeare
  []
  (slurp (shakespeare-file)))

