function ajaxpost () {
  // (A) GET FORM DATA
  var form = document.getElementById("myForm");
  var data = "{
    "type" : "cat",
    "price" : 123.11
}";
 
  // (B) AJAX REQUEST
  // (B1) POST DATA TO SERVER, RETURN RESPONSE AS TEXT
  fetch("https://vyfeovv08c.execute-api.us-east-2.amazonaws.com/beta", { method:"POST", body:data})
  .then(res=>res.text())
 
  // (B2) SHOW MESSAGE ON SERVER RESPONSE
  .then((response) => {
    console.log(response);
    if (response == "Low Risk") { alert("SUCCESSFUL!"); }
    else { alert("FAILURE!"); }
  })
 
  // (B3) OPTIONAL - HANDLE FETCH ERROR
  .catch((err) => { console.error(err); });
 
  // (C) PREVENT FORM SUBMIT
  return false;
}
