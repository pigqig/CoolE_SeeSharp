using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using VisionStudio.Services;

namespace VisionStudio.Pages.Datasets;

public class AnnotateModel : PageModel
{
    private readonly DatasetService _ds;
    public AnnotateModel(DatasetService ds) => _ds = ds;
    public string Id { get; private set; } = "";
    public string Name { get; private set; } = "";

    public IActionResult OnGet(string id)
    {
        var info = _ds.Get(id);
        if (info == null) return RedirectToPage("/Datasets/Index");
        Id = info.Id;
        Name = info.Name;
        return Page();
    }
}
