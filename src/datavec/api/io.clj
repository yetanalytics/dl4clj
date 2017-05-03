(ns dl4clj.datavec.api.io
  (:import [org.datavec.api.io DataInputBuffer DataOutputBuffer]))

;;https://deeplearning4j.org/datavecdoc/org/datavec/api/io/WritableComparator.html
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/io/BinaryComparable.html
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/io/WritableConverter.html
;; users should never have to directly create these ^
;; if I am wrong, come back and implement
;; I do need these for dl4clj.datasets.datavec.clj

(defn new-data-input-buffer
  "A reusable DataInput implementation that reads from an in-memory buffer.
  This saves memory over creating a new DataInputStream and ByteArrayInputStream
  each time data is read.

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/DataInputBuffer.html"
  []
  (DataInputBuffer.))

(defn new-data-output-buffer
  "A reusable DataOutput implementation that writes to an in-memory buffer.
  This saves memory over creating a new DataOutputStream and ByteArrayOutputStream
  each time data is written.

  :size (int) size of the output buffer
   -if not supplied, creates a new empty buffer

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/DataOutputBuffer.html"
  [& {:keys [size]
      :as opts}]
  (if (contains? opts :size)
    (DataOutputBuffer. size)
    (DataOutputBuffer.)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; api interaction fns for buffers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-data
  [buffer]
  (.getData buffer))

(defn get-length
  "returns the length of the input"
  [buffer]
  (.getLength buffer))

(defn get-position
  "returns the current position in the input"
  [buffer]
  (.getPosition buffer))

(defn reset-buffer!
  "resets the data the the buffer reads

  :buffer is a data-buffer
  :input-data (byte-array) the data going into the buffer
  :start (int), starting position within the input-data
  :length (int), length of the input-data"
  [& {:keys [buffer input-data start length]
      :as opts}]
  (if (contains? opts :start)
    (doto buffer (.reset input-data start length))
    (doto buffer (.reset input-data length))))

(defn write!
  "Writes bytes from an input-buffer directly into the output-buffer

  :data-input (java.io.DataInput), some kind of data-input stream/buffer
  :length (int), length of the output
  :output-buffer (output-buffer), an output buffer being written to"
  [& {:keys [data-input length output-buffer]}]
  (doto output-buffer (.write data-input length)))

(defn write-to!
  "write to a file stream given some output buffer
  :output-buffer (output-buffer), an output buffer being written from
  :output-stream (java.io.OutputStream) where the data is being written to"
  [& {:keys [output-buffer output-stream]}]
  (doto output-buffer (.writeTo output-stream)))
