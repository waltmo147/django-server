// Sharon Jeong (SJ632)
$(document).ready(function() {
	//Slideshow*******************************************************************
	var images = ["images/babys_breaths.jpg", "images/buttercups.jpg", "images/carnations.jpg", "images/gardenias.jpg", "images/hydrangeas.jpg", "images/orchids.jpg", "images/peonies.jpg", "images/red_roses.jpg", "images/tulips.jpg", "images/white_roses.jpg"];
	//Images from 1-800-flowers, Gardenerdy and Bloomsbythebox
	var description = ["Baby's Breaths", "Buttercups", "Carnations", "Gardenias", "Hydrangeas", "Orchids", "Peonies", "Red Roses", "Tulips", "White Roses"];

	var index = 0;
	// Change image when the button is clicked
	$("#change").click(function() {
		index=index+1;
		$("#current_img").attr("src", images[index%images.length]);
		$("#current_img").attr("alt", description[index%images.length]);
		$("#caption").text(description[index%images.length]);
	});
});
